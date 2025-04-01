import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.feature_dims * 3, args.feature_dims)
        self.proj2 = nn.Linear(args.feature_dims, 1)

    def forward(self, batch):
        # batch: [batchsize, feature_dims]
        inputs = torch.cat([batch['language'], batch['audio'], batch['video']], dim=-1)
        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output
