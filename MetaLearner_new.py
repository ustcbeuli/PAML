import torch
from torch.nn import functional as F
import numpy as np
from util import *


# social embedding: prediction
class MetaLearner(torch.nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.use_cuda = config['use_cuda']
        self.config = config

        # prediction parameters
        self.vars = torch.nn.ParameterDict()
        self.embd_dim = config['embedding_dim']

        self.social_types = 2 + int(self.config['use_coclick'])
        self.social_encoder = SocialEncoder(config, self.vars, 'social')
        self.vars['social_merge'] = self.get_initialed_para_matrix(1, self.social_types)
        
        self.fc1_in_dim = self.embd_dim * (config['use_fea_item'] + 2)
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.dropout = config['dropout']

        self.vars['ml_fc_w1'] = self.get_initialed_para_matrix(self.fc2_in_dim, self.fc1_in_dim)
        self.vars['ml_fc_b1'] = self.get_zero_para_bias(self.fc2_in_dim)

        self.vars['ml_fc_w2'] = self.get_initialed_para_matrix(self.fc2_out_dim, self.fc2_in_dim)
        self.vars['ml_fc_b2'] = self.get_zero_para_bias(self.fc2_out_dim)

        self.vars['ml_fc_w3'] = self.get_initialed_para_matrix(1, self.fc2_out_dim)
        self.vars['ml_fc_b3'] = self.get_zero_para_bias(1)

    def get_initialed_para_matrix(self, out_num, in_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))

    def forward(self, item_emb, user_emb, vars_dict=None, **kwargs):
        """
        """
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_emb

        preference = self.get_user_social_preference(vars_dict, **kwargs)
        x_self = preference[0].repeat(x_i.shape[0], 1)
        x_social = preference[1].repeat(x_i.shape[0], 1)
        x = torch.cat((x_i, x_self, x_social), 1)

        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3'])
        return x.squeeze()

    def get_task_embd(self, **kwargs):
        preference = self.get_user_social_preference(self.vars, **kwargs)
        task_embd = torch.cat(preference, dim=1)
        return task_embd

    def get_user_social_preference(self, vars_dict=None, **kwargs):
        if vars_dict is None:
            vars_dict = self.vars

        self_users, self_items, self_mask = kwargs['self_users_embd'], kwargs['self_items_embd'], kwargs['self_mask']
        x_self_list, x_social_list = [], []

        social_users, social_items, social_mask = kwargs['social_users_embd'], kwargs['social_items_embd'], kwargs['social_mask']
        x_self, x_social = self.social_encoder(self_users, self_items, self_mask, social_users, social_items, social_mask, vars_dict)
        x_self_list.append(x_self)
        x_social_list.append(x_social)

        implicit_users, implicit_items, implicit_mask = kwargs['implicit_users_embd'], kwargs['implicit_items_embd'], kwargs['implicit_mask']
        x_self, x_social = self.social_encoder(self_users, self_items, self_mask, implicit_users, implicit_items, implicit_mask, vars_dict)
        x_self_list.append(x_self)
        x_social_list.append(x_social)

        if self.config['use_coclick']:
            coclick_users, coclick_items, coclick_mask = kwargs['coclick_users_embd'], kwargs['coclick_items_embd'], kwargs['coclick_mask']
            x_self, x_social = self.social_encoder(self_users, self_items, self_mask, coclick_users, coclick_items, coclick_mask, vars_dict)
            x_self_list.append(x_self)
            x_social_list.append(x_social)

        x_self = x_self_list[0]
        x_social = torch.cat(x_social_list, 0)
        x_social = torch.mm(vars_dict['social_merge'], x_social)

        return x_self, x_social


    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars

    def get_parameter_size(self):
        parameter_name_size = dict()
        for key in self.vars.keys():
            weight_size = np.prod(self.vars[key].size())
            parameter_name_size[key] = weight_size

        return parameter_name_size

    def get_user_preference_embedding(self, items_emb, users_embd, mask):
        """
        :param items_emb:
        :param users_embd:
        :param mask:
        :return:
        """
        # users_items_embd = torch.cat((users_embd, items_emb), dim=-1)
        # users_items_embd = users_items_embd * torch.unsqueeze(mask, dim=2)
        # users_items_embd = torch.sum(users_items_embd, dim=1)
        # mask_len = torch.sum(mask, dim=1, keepdim=True)
        # users_items_embd = torch.div(users_items_embd, mask_len)
        # user_preference_embd = F.relu(F.linear(users_items_embd, self.vars['ml_user_w'], self.vars['ml_user_b']))
        item_embd_agg = items_emb * torch.unsqueeze(mask, dim=2)
        item_embd_agg = torch.sum(item_embd_agg, dim=1)
        mask_len = torch.sum(mask, dim=1, keepdim=True)
        user_preference_embd = torch.div(item_embd_agg, mask_len)
        return user_preference_embd


# social embedding: embedding
class SocialEncoder(torch.nn.Module):
    def __init__(self, config, vars_dict, name):
        super(SocialEncoder, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.vars = vars_dict
        self.name = name
        self.embd_dim = config['embedding_dim']
        used_feat_num = config['use_fea_user'] + config['use_fea_item']
        self.vars['ml_user_w'] = self.get_initialed_para_matrix(self.embd_dim, self.embd_dim * used_feat_num)
        self.vars['ml_user_b'] = self.get_zero_para_bias(self.embd_dim)
        self.social_types = 2 + int(self.config['use_coclick'])
        self.vars['ml_social_w1'] = self.get_initialed_para_matrix(self.embd_dim, self.embd_dim)
        self.vars['ml_social_b1'] = self.get_zero_para_bias(self.embd_dim)
        self.vars['ml_social_w2'] = self.get_initialed_para_matrix(1, self.embd_dim)

    def forward(self, self_users, self_items, self_mask, social_users, social_items, social_mask, vars_dict=None):
        """
        """
        if vars_dict is None:
            vars_dict = self.vars
        self_embd_agg = self.aggregate_items(self_users, self_items, self_mask)
        self_embd = F.relu(F.linear(self_embd_agg, vars_dict['ml_user_w'], vars_dict['ml_user_b']))
        if self.social_types == 0:
            self_embd = self.aggregate_rate(torch.squeeze(self_embd, dim=1), vars_dict)
            return self_embd

        if social_users is None:
            social_embd = self_embd
        else:
            social_embd_agg = self.aggregate_items(social_users, social_items, social_mask)
            social_embd = F.relu(F.linear(social_embd_agg, vars_dict['ml_user_w'], vars_dict['ml_user_b']))

        att = torch.bmm(self_embd, social_embd.transpose(1, 2)).softmax(dim=2)
        social_embd = torch.bmm(att, social_embd)

        self_embd = self.aggregate_rate(torch.squeeze(self_embd, dim=1), vars_dict)
        social_embd = self.aggregate_rate(torch.squeeze(social_embd, dim=1), vars_dict)

        return self_embd, social_embd

    def aggregate_rate(self, embd, vars_dict):
        embd_trans = torch.relu(F.linear(embd, vars_dict['ml_social_w1'], vars_dict['ml_social_b1']))
        att = torch.mm(vars_dict['ml_social_w2'], embd_trans.transpose(0, 1)).softmax(dim=1)
        embd = torch.mm(att, embd)
        return embd

    def aggregate_items(self, users, items, mask):
        users_items = torch.cat((users, items), dim=-1) * torch.unsqueeze(mask, dim=3)
        users_items_sum = torch.sum(users_items, dim=2)
        mask_len = torch.sum(mask, dim=2, keepdim=True) + 0.0001
        users_items_agg = torch.div(users_items_sum, mask_len)
        return users_items_agg

    def get_initialed_para_matrix(self, out_num, in_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w

    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars