# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function
import argparse
import base64
import logging
import os.path as op
import os
import random
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import logging
import sys

sys.path.insert(0, '..')
from oscar.utils.misc import (mkdir, set_seed)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader
from oscar.utils.cbs import FiniteStateMachineBuilder
from transformers.pytorch_transformers import BertTokenizer, BertConfig


LABEL2ID = {
    'contradiction': 0,
    'neutral': 1,
    'entailment': 2
}
ID2LABEL = {
    0: 'contradiction',
    1: 'neutral',
    2: 'entailment'
}
SPECIAL_TOKENS_DICT = {
    'additional_special_tokens': ['[premise]', '[hypothesis]', '[expl]']
}


def get_data(data_type, args):
    data = {}
    data['expl'] = [line.rstrip() for line in open(
        os.path.join(args.data_dir, f"expl_1.{data_type}"), 'r')]
    data['label'] = [line.rstrip() for line in open(
        os.path.join(args.data_dir, f"labels.{data_type}"), 'r')]
    data['label_int'] = [
        LABEL2ID[i] for i in data['label']]
    data['hypothesis'] = [line.rstrip() for line in open(
        os.path.join(args.data_dir, f"s2.{data_type}"), 'r')]
    data['image_id'] = [line.rstrip() for line in open(
        os.path.join(args.data_dir, f"images.{data_type}"), 'r')]
    data['all_image_ids'] = torch.load(
        os.path.join(args.image_data_dir, 'flickr30k_vesnli_vg_detectron_img_names.pt'))
    data['image_bbox'] = torch.load(
        os.path.join(args.image_data_dir, 'flickr30k_vesnli_vg_detectron_boxes.pt'))
    data['image_obj_lbl_ids'] = torch.load(
        os.path.join(args.image_data_dir, 'flickr30k_vesnli_vg_detectron_obj_lbl_ids.pt'))
    # data['image_feat'] = torch.load(
    #     os.path.join(args.image_data_dir, 'flickr30k_vesnli_vg_detectron_feat.pt'))
    return data


class CaptionTSVDataset(Dataset):
    def __init__(self, data_type, tokenizer, args, is_train):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.data = get_data(data_type, args)
        self.tokenizer = tokenizer
        self.tensorizer = CaptionTensorizer(self.tokenizer, args,
                                            is_train=is_train)
        self.args = args
        self.is_train = is_train

        # Load VG Classes
        self.vg_classes = ['__background__']
        with open(os.path.join(args.image_data_dir, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.vg_classes.append(object.split(',')[0].lower().strip())

    def generate_text_a(self, idx):
        add_spl_tkn = SPECIAL_TOKENS_DICT['additional_special_tokens']

        hypothesis = add_spl_tkn[1] + self.data['hypothesis'][idx]
        if self.is_train:
            expl = add_spl_tkn[2] + self.data['expl'][idx]
        else:
            expl = add_spl_tkn[2]
        text_a = hypothesis + expl
        return text_a

    def get_image_global_index(self, idx):
        image_id = self.data['image_id'][idx]
        return self.data['all_image_ids'].index(image_id)

    def get_od_labels(self, img_idx):
        obj_labels = self.data['image_obj_lbl_ids'][img_idx]
        obj_labels = [self.vg_classes[i] for i in obj_labels]
        return " ".join(list(set(obj_labels)))

    def get_image_features(self, img_idx):
        img_id_name = self.data['all_image_ids'][img_idx]
        feat = torch.load(os.path.join(self.args.image_data_dir,
                                       f'features/{img_id_name[:-4]}_feat.pt'))
        # feat = self.data['image_feat'][img_idx]
        bbox = self.data['image_bbox'][img_idx]
        pos_vec = torch.stack(
            [abs(bbox[:, 0] - bbox[:, 2]), abs(bbox[:, 1] - bbox[:, 3])]).reshape(-1, 2)
        return torch.cat([feat, bbox, pos_vec], dim=1)

    def get_labels(self, idx):
        return self.data['label_int'][idx]

    def __getitem__(self, idx):
        img_idx = self.get_image_global_index(idx)
        # print('img_idx', img_idx)
        # features = self.data['image_feat'][img_idx]
        input_text = self.generate_text_a(idx)
        label = self.get_labels(idx)
        features = self.get_image_features(img_idx)
        od_labels = self.get_od_labels(img_idx)
        example = self.tensorizer.tensorize_example(
            input_text, features, text_b=od_labels, labels=label)
        return img_idx, example

    def __len__(self):
        return len(self.data['label'])


class CaptionTensorizer(object):
    def __init__(self, tokenizer, args, is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            is_train: train or test mode.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = args.max_img_seq_length
        self.max_seq_len = args.max_seq_length
        self.max_seq_a_len = args.max_seq_a_length
        self.mask_prob = args.mask_prob
        self.max_masked_tokens = args.max_masked_tokens
        self.args = args
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
                                                     self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None, labels=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1):

        tokens_a = self.tokenizer.tokenize(text_a)
        if not self.is_train:
            # fake tokens to generate masks
            tokens_a += [self.tokenizer.mask_token] * \
                (self.max_seq_a_len - 2 - len(tokens_a))
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + \
            tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + \
            [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(
                range(1, seq_a_len))  # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = int(min(max(round(self.mask_prob * seq_a_len),
                                     1),
                                 self.max_masked_tokens))
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                                               (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            mask_pos_id = tokens.index(self.tokenizer.mask_token)
            masked_pos = torch.tensor(
                [0] * mask_pos_id + [1] * (self.max_seq_len - mask_pos_id)).int()
            # masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        elif img_len < self.max_img_seq_len:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start: c_end, c_start: c_end].copy_(
            self._triangle_mask[0: seq_a_len, 0: seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        # full attention for L-R:
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids, labels)
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, labels)


if __name__ == "__main__":
    from seq_clf_gen_vesnli_args import get_args
    args = get_args()
    if args.do_train:
        data_type = 'train'
    if args.do_test:
        data_type = 'test'
    if args.do_eval:
        data_type = 'dev'

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                              else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    dataset = CaptionTSVDataset(data_type,
                                tokenizer=tokenizer,
                                is_train=args.do_train,
                                args=args)
    dataloader = DataLoader(dataset, batch_size=16)

    batch = next(iter(dataloader))
    img_keys, batch = batch
    if args.do_train:
        input_ids, attention_mask, token_type_ids, img_feats, masked_pos, masked_ids, labels = batch
    else:
        input_ids, attention_mask, token_type_ids, img_feats, masked_pos, labels = batch

    # print('img_keys', img_keys)
    print('input_ids', input_ids.shape, input_ids[0])
    print('input_ids', tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
    if args.do_train:
        print('masked_ids', masked_ids.shape, masked_ids[0])
        print('masked_ids', tokenizer.convert_ids_to_tokens(
            masked_ids[0].tolist()))
    print('masked_pos', masked_pos.shape, masked_pos[0])
    print('attention_mask', attention_mask.shape, attention_mask[0])
    print('token_type_ids', token_type_ids.shape, token_type_ids[0])
    print(token_type_ids[0].tolist().count(0),
          token_type_ids[0].tolist().count(1))
    print('img_feats', img_feats.shape)
    print('labels', labels.shape, labels)
