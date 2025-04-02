import torch
from torch import nn
import torch.nn.functional as F


missing_type_index = {'language': 1, 'video': 2, 'audio': 3}


class modal_sum(nn.Module):
    def __init__(self, args):
        super(modal_sum, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim, args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            data = self.modal_proj[modal](batch[modal])
            data[missing_index == missing_type_index[modal]] = torch.zeros_like(data[missing_index == missing_type_index[modal]])
            inputs.append(data)
        inputs = sum(inputs)

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output


class modal_concat_zero_padding(nn.Module):
    def __init__(self, args):
        super(modal_concat_zero_padding, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim * len(args.modality_types), args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            batch[modal][missing_index == missing_type_index[modal]] = torch.zeros_like(batch[modal][missing_index == missing_type_index[modal]])
            inputs.append(self.modal_proj[modal](batch[modal]))
        inputs = torch.cat(inputs, dim=-1)

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output
