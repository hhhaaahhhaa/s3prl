# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import PitchDataset
from argparse import Namespace
from pathlib import Path

SAMPLE_RATE = 16000


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        root_dir = Path(self.datarc['file_path'])

        self.train_dataset = PitchDataset('train', root_dir, self.datarc['meta_data'], upstream_rate=upstream_rate)
        self.dev_dataset = PitchDataset('dev', root_dir, self.datarc['meta_data'], upstream_rate=upstream_rate)
        self.test_dataset = PitchDataset('test', root_dir, self.datarc['meta_data'], upstream_rate=upstream_rate)
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = 1,
            **model_conf,
        )
        self.loss_func = SimpleMSELoss()
        # self.objective = nn.CrossEntropyLoss()
        # self.register_buffer('best_score', torch.zeros(1))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=False, sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )
        # (sampler is None)

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features])
        labels = [feat[:l] for feat, l in zip(labels, features_len)]

        # Sequence mask
        max_len = max(features_len)
        mask = torch.arange(max_len).expand(len(features_len), max_len) < features_len.unsqueeze(1)

        features_len = features_len.to(device=device)
        mask = mask.to(device=device)

        features = pad_sequence(features, batch_first=True)
        labels = pad_sequence(labels, batch_first=True).to(device=device)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        # print(mask.shape)
        # print(features_len)
        # print(predicted.shape, labels.shape)
        # assert 1 == 2

        loss = self.loss_func(predicted, labels, mask.unsqueeze(2))

        records['loss'].append(loss.item())

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        pass
    #     save_names = []
    #     for key in ["acc", "loss"]:
    #         average = torch.FloatTensor(records[key]).mean().item()
    #         logger.add_scalar(
    #             f'voxceleb1/{mode}-{key}',
    #             average,
    #             global_step=global_step
    #         )
    #         with open(Path(self.expdir) / "log.log", 'a') as f:
    #             if key == 'acc':
    #                 print(f"{mode} {key}: {average}")
    #                 f.write(f'{mode} at step {global_step}: {average}\n')
    #                 if mode == 'dev' and average > self.best_score:
    #                     self.best_score = torch.ones(1) * average
    #                     f.write(f'New best on {mode} at step {global_step}: {average}\n')
    #                     save_names.append(f'{mode}-best.ckpt')

    #     if mode in ["dev", "test"]:
    #         with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
    #             lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_speaker"])]
    #             file.writelines(lines)

    #         with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
    #             lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth_speaker"])]
    #             file.writelines(lines)

    #     return save_names


class SimpleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pitch_predictions, pitch_targets, mask):
        pitch_predictions = pitch_predictions.masked_select(mask)
        pitch_targets = pitch_targets.masked_select(mask)
        pitch_loss = self.loss(pitch_predictions, pitch_targets)
        return pitch_loss
