import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Task_specific(torch.nn.Module):
    def __init__(self, config, parameter_name_size):
        super(Task_specific, self).__init__()
        self.config = config
        self.parameter_name_size = parameter_name_size
        self.in_fea = config['embedding_dim'] + config['embedding_dim']
              
        all_para_size = 0
        for name, size in parameter_name_size.items():
            all_para_size += size
        self.fc = torch.nn.Linear(in_features=self.in_fea, out_features=all_para_size)

    def forward(self, task_embedding, ml_weights):
        task_weights = dict()
        fc_out = torch.sigmoid(self.fc(task_embedding))
        start_idx = 0
        for name, size in self.parameter_name_size.items():
            para = ml_weights[name]
            task_weights[name] = para * torch.reshape(fc_out[0, start_idx: start_idx + size], para.size())
            start_idx += size
        return task_weights
