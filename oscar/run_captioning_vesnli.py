# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
'''TRAIN
python oscar/run_captioning_vesnli.py     \
    --do_train     \
    --do_lower_case     \
    --add_od_labels     \
    --model_name_or_path pretrained_models/base-vg-labels/ep_67_588997     \
    --data_dir=/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE     \
    --max_seq_a_length=80     \
    --max_seq_length=110      \
    --learning_rate 0.00003     \
    --per_gpu_train_batch_size 16 \
    --gradient_accumulation_step=4     \
    --num_train_epochs 30     \
    --save_steps 5000


TEST
python oscar/run_captioning_vesnli.py \
    --do_test \
    --data_dir=/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE     \
    --per_gpu_eval_batch_size 16 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir /home/hdd1/vibhav/VE-SNLI/Oscar/output/10_13_20_25_18/checkpoint-6-50000
'''

from __future__ import absolute_import, division, print_function
import argparse
import base64
from datetime import datetime
import os
import os.path as op
import random
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset, DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from oscar.utils.logger import setup_logger
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import tsv_writer
from oscar.utils.misc import (mkdir, set_seed,
                              load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption,
                                          evaluate_on_nocaps, ScstRewardCriterion)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader
from oscar.utils.cbs import FiniteStateMachineBuilder
from oscar.modeling.modeling_bert import BertForImageCaptioning
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from run_captioning_dataset_vesnli import *
from run_captioning_args_vesnli import get_args


def build_dataset(data_type, tokenizer, args, is_train=True):
    dataset = CaptionTSVDataset(data_type,
                                tokenizer=tokenizer,
                                is_train=is_train,
                                args=args)
    if args.small_data != -1:
        logger.info('Using small subset of data')
        dataset = Subset(dataset, list(range(args.small_data)))
    return dataset


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return checkpoint_dir


def train(args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) //
                                                   args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
            * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not
                    any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
            optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                          args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if args.scst:
        scst_criterion = ScstRewardCriterion()
        logger.info("  SCST training...")

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, (img_keys, batch) in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)

            if not args.scst:
                model.train()
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                          'token_type_ids': batch[2], 'img_feats': batch[3],
                          'masked_pos': batch[4], 'masked_ids': batch[5]
                          }
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                masked_ids = inputs['masked_ids']
                masked_ids = masked_ids[masked_ids != 0]

                logits = torch.max(logits, -1)[1].data  # argmax
                batch_score = logits == masked_ids

                batch_acc = torch.sum(batch_score.float()) / \
                    torch.sum(inputs['masked_pos'])
            else:
                loss = scst_train_iter(
                    args, train_dataset, model, scst_criterion, img_keys, batch, tokenizer)
                batch_acc = scst_criterion.get_score()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), "
                                "score: {:.4f} ({:.4f})".format(epoch, global_step,
                                                                optimizer.param_groups[0]["lr"], loss, global_loss /
                                                                global_step,
                                                                batch_acc, global_acc / global_step)
                                )
                    if args.local_rank == 0:
                        torch.distributed.barrier()

                    # writer.add_scalar('Loss', loss, global_step)
                    # writer.add_scalar('Acc', global_acc /
                    #                   global_step, global_step)

                if (args.local_rank in [-1, 0]) and (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                    checkpoint_dir = save_checkpoint(
                        model, tokenizer, args, epoch, global_step)
                    # # evaluation
                    # if args.evaluate_during_training:
                    #     logger.info("Perform evaluation at step: %d" %
                    #                 (global_step))
                    #     evaluate_file = evaluate(args, val_dataset, model, tokenizer,
                    #                              checkpoint_dir)
                    #     with open(evaluate_file, 'r') as f:
                    #         res = json.load(f)
                    #     best_score = max(best_score, res['CIDEr'])
                    #     res['epoch'] = epoch
                    #     res['global_step'] = step
                    #     res['best_CIDEr'] = best_score
                    #     eval_log.append(res)
                    #     with open(args.output_dir + '/eval_logs.json', 'w') as f:
                    #         json.dump(eval_log, f)
    return global_step, global_loss / global_step


def scst_train_iter(args, train_dataset, model, scst_criterion, img_keys, batch, tokenizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
         tokenizer.mask_token]
    )
    inputs = {'is_decode': True,
              'input_ids': batch[0], 'attention_mask': batch[1],
              'token_type_ids': batch[2], 'img_feats': batch[3],
              'masked_pos': batch[4],
              'do_sample': False,
              'bos_token_id': cls_token_id,
              'pad_token_id': pad_token_id,
              'eos_token_ids': [sep_token_id, pad_token_id],
              'mask_token_id': mask_token_id,
              # for adding od labels
              'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

              # hyperparameters of beam search
              'max_length': args.max_seq_a_length,
              'num_beams': 1,
              "temperature": args.temperature,
              "top_k": args.top_k,
              "top_p": args.top_p,
              "repetition_penalty": args.repetition_penalty,
              "length_penalty": args.length_penalty,
              "num_return_sequences": 1,
              "num_keep_best": 1,
              }

    model.eval()
    with torch.no_grad():
        greedy_res_raw, _ = model(**inputs)
        greedy_res_raw.squeeze_(1)  # batch_size * max_len

    model.train()
    inputs['do_sample'] = True
    sample_res_raw, sample_logprobs = model(**inputs)
    sample_res_raw.squeeze_(1)
    sample_logprobs.squeeze_(1)
    assert sample_logprobs.requires_grad == True
    assert sample_res_raw.requires_grad == False

    def _ids_to_captions(all_ids):
        captions = []
        for ids in all_ids:
            c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            captions.append(c)
        return captions

    greedy_res = _ids_to_captions(greedy_res_raw)
    sample_res = _ids_to_captions(sample_res_raw)
    gt_res = [train_dataset.get_captions_by_key(k) for k in img_keys]

    loss = scst_criterion(gt_res, greedy_res, sample_res, sample_logprobs)
    return loss


def get_predict_file(data_type, output_dir, args):
    cc = ['generated']
    # make sure it works with/without / in end of the path.
    data = op.basename(op.join(args.data_dir, '')[:-1])
    cc.append(data)
    cc.append(data_type)
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.add_od_labels:
        cc.append('odlabels')
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.use_cbs:
        cc.append('cbs{}'.format(args.min_constraints_to_satisfy))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.csv'.format('.'.join(cc)))


# '''
# NOT TO  BE USED
# '''
def evaluate(args, val_dataset, model, tokenizer, output_dir):
    assert op.isdir(output_dir)
    predict_file = get_predict_file('dev', output_dir, args)
    if op.isfile(predict_file):
        logger.info('Skip predict. {} already exists'.format(predict_file))
    else:
        test(args, val_dataset, model, tokenizer, predict_file)

    assert predict_file.endswith('.tsv')
    evaluate_file = op.splitext(predict_file)[0] + '.eval.json'

    if op.isfile(evaluate_file):
        logger.info('Skip evaluation. {} already exists'.format(evaluate_file))
        return evaluate_file

    eval_method = 'nocaps' if 'nocaps' in op.basename(predict_file) else 'coco'
    if eval_method == 'coco':
        gt_file = val_dataset.get_caption_file_in_coco_format()
        result = evaluate_on_coco_caption(
            predict_file, gt_file, outfile=evaluate_file)
    else:
        split = 'val' if 'val' in op.basename(
            val_dataset.yaml_file) else 'test'
        result = evaluate_on_nocaps(split, predict_file,
                                    data_dir=args.data_dir, evaluate_file=evaluate_file)
    logger.info("evaluation result: {}".format(str(result)))
    return evaluate_file


def test(args, test_dataset, model, tokenizer, predict_file):
    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(
        test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.test_batch_size, num_workers=args.num_workers)

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token,
                                         tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, '.'])
    model.eval()

    # # restore existing results for long running inference tasks
    # exist_key2pred = {}
    # tmp_file = predict_file + '.tmp.copy'
    # if op.isfile(tmp_file):
    #     with open(tmp_file, 'r') as fp:
    #         for line in fp:
    #             parts = line.strip().split('\t')
    #             if len(parts) == 2:
    #                 exist_key2pred[parts[0]] = parts[1]

    columns = ['model_input', 'expl', 'gen_expl']
    with torch.no_grad():
        for step, (img_keys, batch) in tqdm(enumerate(test_dataloader)):
            # is_exist = True
            # for k in img_keys:
            #     if k not in exist_key2pred:
            #         is_exist = False
            #         break
            # if is_exist:
            #     for k in img_keys:
            #         yield k, exist_key2pred[k]
            #     continue
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'is_decode': True,
                      'input_ids': batch[0], 'attention_mask': batch[1],
                      'token_type_ids': batch[2], 'img_feats': batch[3],
                      'masked_pos': batch[4],
                      'do_sample': False,
                      'bos_token_id': cls_token_id,
                      'pad_token_id': pad_token_id,
                      'eos_token_ids': [sep_token_id, pad_token_id],
                      'mask_token_id': mask_token_id,
                      # for adding od labels
                      'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

                      # hyperparameters of beam search
                      'max_length': args.max_gen_length,
                      'num_beams': args.num_beams,
                      "temperature": args.temperature,
                      "top_k": args.top_k,
                      "top_p": args.top_p,
                      "repetition_penalty": args.repetition_penalty,
                      "length_penalty": args.length_penalty,
                      "num_return_sequences": args.num_return_sequences,
                      "num_keep_best": args.num_keep_best,
                      }
            if args.use_cbs:
                inputs.update({'use_cbs': True,
                               'fsm': batch[5],
                               'num_constraints': batch[6],
                               'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
                               })
            # captions, logprobs
            outputs = model(**inputs)
            all_generated = outputs[0]  # batch_size * num_keep_best * max_len
            for img_key, input_ids, generated in zip(img_keys, batch[0], all_generated):
                print('INPUT: ', tokenizer.decode(input_ids.tolist()))
                print('GEN.: ', tokenizer.decode(generated.tolist()[0]))
                # res = []
                # for cap, conf in zip(generated, confs):
                #     cap = tokenizer.decode(
                #         cap.tolist(), skip_special_tokens=True)
                #     res.append({'caption': cap, 'conf': conf.item()})
                # if isinstance(img_key, torch.Tensor):
                #     img_key = img_key.item()


def restore_training_settings(args):
    assert not args.do_train
    assert args.do_test or args.do_eval
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
            max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
                       'max_img_seq_length', 'img_feature_dim',
                       'img_feature_type']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                                                                              test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    args = get_args()

    global logger
    # global logger, writer

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    if args.do_train:
        mkdir(args.output_dir)

        t = datetime.today()
        args.output_dir = op.join(args.output_dir,
                                  f"{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}")
        if not op.exists(args.output_dir):
            mkdir(args.output_dir)

        logger = setup_logger("vlpretrain", args.output_dir, args.local_rank)
    else:
        logger = setup_logger(
            "vlpretrain", os.path.dirname(args.eval_model_dir), args.local_rank, 'test_log.txt')

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    set_seed(args.seed, args.n_gpu)

    # writer = SummaryWriter(log_dir=args.output_dir, flush_secs=60)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    if args.do_train:
        assert args.model_name_or_path is not None
        config = config_class.from_pretrained(args.config_name if args.config_name else
                                              args.model_name_or_path, num_labels=args.num_labels, finetuning_task='image_captioning')
        if args.scst:
            # avoid using too much memory
            config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                                    else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        assert op.isdir(args.eval_model_dir)
        config = config_class.from_pretrained(args.eval_model_dir)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(args.eval_model_dir)
        logger.info("Evaluate the following checkpoint: %s",
                    args.eval_model_dir)
        model = model_class.from_pretrained(args.eval_model_dir, config=config)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = build_dataset('train', tokenizer, args)
        val_dataset = build_dataset('dev', tokenizer, args, is_train=False)
        global_step, avg_loss = train(
            args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s",
                    global_step, avg_loss)

    # # inference and evaluation
    # if args.do_test or args.do_eval:
    #     args = restore_training_settings(args)
    #     test_dataset = build_dataset('test', tokenizer, args, is_train=False)
    #     if args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)

    #     if not args.do_eval:
    #         predict_file = get_predict_file('test', args.eval_model_dir, args)
    #         test(args, test_dataset, model, tokenizer, predict_file)
    #         logger.info("Prediction results saved to: {}".format(predict_file))
    #     else:
    #         evaluate_file = evaluate(args, test_dataset, model, tokenizer,
    #                                  args.eval_model_dir)
    #         logger.info(
    #             "Evaluation results saved to: {}".format(evaluate_file))

    if args.do_test and args.local_rank in [-1, 0]:
        args = restore_training_settings(args)
        test_dataset = build_dataset('test', tokenizer, args, is_train=False)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        predict_file = get_predict_file('test', args.eval_model_dir, args)
        test(args, test_dataset, model, tokenizer, predict_file)
        logger.info("Prediction results saved to: {}".format(predict_file))

    if args.do_eval and args.local_rank in [-1, 0]:
        args = restore_training_settings(args)
        dev_dataset = build_dataset('dev', tokenizer, args, is_train=False)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        predict_file = get_predict_file('dev', args.eval_model_dir, args)
        test(args, dev_dataset, model, tokenizer, predict_file)
        logger.info("Prediction results saved to: {}".format(predict_file))


if __name__ == "__main__":
    main()
