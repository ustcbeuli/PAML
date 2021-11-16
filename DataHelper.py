import gc
import glob
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from util import *
import random
import collections


class DataHelper:
    def __init__(self, config, states):
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.input_dir = config['input_dir']
        self.uid_feat_dict = dict()
        self.data = dict()
        for state in states:
            self.data[state] = self.load_data(state)
        self.sample_users = []
        self.sample_uid_feat_tensor_dict = None
        self.implicit_neighbor_matrix = None

    def load_data(self, state):
        data_dir = self.input_dir
        device = self.device
        supp_x_list = pickle.load(open("{}/{}/supp_x.pkl".format(data_dir, state), "rb"))
        supp_y_list = pickle.load(open("{}/{}/supp_y.pkl".format(data_dir, state), "rb"))
        query_x_list = pickle.load(open("{}/{}/query_x.pkl".format(data_dir, state), "rb"))
        query_y_list = pickle.load(open("{}/{}/query_y.pkl".format(data_dir, state), "rb"))
        social_info_list = pickle.load(open("{}/{}/social_info.pkl".format(data_dir, state), "rb"))
        supp_xs_s, supp_ys_s, query_xs_s, query_ys_s = [], [], [], []
        task_self_s, task_social_s, task_implicit_s, task_coclick_s = [], [], [], []

        if state == 'meta_training':
            for supp_x in supp_x_list:
                self.uid_feat_dict[supp_x['uid']] = supp_x['feat_list']
        social_zero_count, social_total_count = 0, 0
        implicit_zero_count, implicit_total_count = 0, 0
        coclick_zero_count, coclick_total_count = 0, 0

        for supp_x, supp_y, query_x, query_y, social_info in zip(supp_x_list, supp_y_list, query_x_list, query_y_list, social_info_list):
            assert supp_x['uid'] == query_x['uid']

            supp_xs_s.append(supp_x['feat_list'].to(device))
            supp_ys_s.append(supp_y.to(device))

            query_xs_s.append(query_x['feat_list'].to(device))
            query_ys_s.append(query_y.to(device))


            self_feat, self_mask = social_info['self_feat'], social_info['self_mask']
            task_self_s.append([self_feat.to(device), self_mask.to(device)])

            social_feat, social_mask = social_info['social_feat'], social_info['social_mask']
            if 'social_num' in self.config and len(social_feat) > 0:
                social_num = self.config['social_num']
                social_feat = social_feat[:, :social_num, :, :]
                social_mask = social_mask[:, :social_num, :]
            task_social_s.append([social_feat.to(device), social_mask.to(device)])

            if len(social_feat) == 0:
                social_zero_count += 1
            else:
                social_total_count += len(social_feat[1])

            implicit_feat, implicit_mask = social_info['implicit_feat'], social_info['implicit_mask']
            implicit_num = self.config['implicit_num']
            if len(implicit_feat) > 0:
                implicit_feat = implicit_feat[:, :implicit_num, :, :]
                implicit_mask = implicit_mask[:, :implicit_num, :]
            task_implicit_s.append([implicit_feat.to(device), implicit_mask.to(device)])

            if len(implicit_feat) == 0:
                implicit_zero_count += 1
            else:
                implicit_total_count += len(implicit_feat[1])

            if self.config['use_coclick']:
                coclick_feat, coclick_mask = social_info['coclick_feat'], social_info['coclick_mask']
                coclick_num = self.config['coclick_num']
                if len(coclick_feat) > 0:
                    coclick_feat = coclick_feat[:, :coclick_num, :, :]
                    coclick_mask = coclick_mask[:, :coclick_num, :]
                task_coclick_s.append([coclick_feat.to(device), coclick_mask.to(device)])

                if len(coclick_feat) == 0:
                    coclick_zero_count += 1
                else:
                    coclick_total_count += len(coclick_feat[1])
            else:
                task_coclick_s.append(None)
 
        print('{}, #support set: {}, #query set: {}'.format(state, len(supp_xs_s), len(query_xs_s)))
        print('user num with zero neighbor: %s; avg num of neighbor: %s' % (social_zero_count, social_total_count / len(supp_x_list)))
        print('user num with zero implicit: %s; avg num of implicit: %s' % (implicit_zero_count, implicit_total_count / len(supp_x_list)))
        print('user num with zero coclick: %s; avg num of coclick: %s' % (coclick_zero_count, coclick_total_count / len(supp_x_list)))
        total_data = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, task_self_s, task_social_s, task_implicit_s, task_coclick_s))
        del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, task_self_s, task_social_s, task_implicit_s, task_coclick_s)
        gc.collect()
        return total_data

    def get_neighbor_tensor(self, user_set):
        # uid_list = [uid for uid in user_set if uid in self.uid_feat_dict]
        uid_list = list(user_set)
        if len(uid_list) == 0:
            return torch.tensor([]), torch.tensor([])
        feat_count = []
        for uid in uid_list:
            feat_count.append(len(self.uid_feat_dict[uid]))
        max_feat_num = max(feat_count)
        feat_len = self.uid_feat_dict[uid_list[0]].shape[1]
        feat_tensor = torch.zeros(len(uid_list), max_feat_num, feat_len, dtype=torch.int32)
        feat_mask_tensor = torch.zeros(len(uid_list), max_feat_num, dtype=torch.float32)
        for idx, uid in enumerate(uid_list):
            feat_tensor[idx, :feat_count[idx], :] = self.uid_feat_dict[uid]
            feat_mask_tensor[idx, :feat_count[idx]] = 1
        return feat_tensor, feat_mask_tensor

    def get_batch(self, state, index, get_all=False):
        if not get_all:
            batch_size = self.config['batch_size']
            data = self.data[state][batch_size * index:batch_size * (index + 1)]
        else:
            data = self.data[state]
        uids = [],
        supp_xs, supp_ys = [], []
        query_xs, query_ys = [], []
        task_self_s, task_social_s, task_implicit_s, task_coclick_s = [], [], [], []
        for sample in data:
            supp_xs.append(sample[0])
            supp_ys.append(sample[1])
            query_xs.append(sample[2])
            query_ys.append(sample[3])
            task_self_s.append(sample[4])
            task_social_s.append(sample[5])
            task_implicit_s.append(sample[6])
            task_coclick_s.append(sample[7])

        return {'supp_xs': supp_xs,
                'supp_ys': supp_ys,
                'query_xs': query_xs,
                'query_ys': query_ys,
                'task_self_s': task_self_s,
                'task_social_s': task_social_s,
                'task_implicit_s': task_implicit_s,
                'task_coclick_s': task_coclick_s,
                }

    def sampling_users(self, state, model):
        # print('%s; start sampling users for %s...' % (get_current_time(), state))
        batch_size = 4220
        if self.config['sample_mode'] in {'pre_dis_top', 'pre_dis'} and self.implicit_neighbor_matrix is None:
            self.implicit_neighbor_matrix = pickle.load(open(self.config['pre_dis_path'], 'rb'))
            for idx in range(len(self.implicit_neighbor_matrix)):
                if idx not in self.uid_feat_dict:
                    self.implicit_neighbor_matrix[:, idx] = 0

        self.sample_users = []
        # total_count = 0
        sample_num = self.config['sample_num']
        uid_list, feat_list = [], []
        # uid_sample_count = collections.defaultdict(int)
        for idx, sample in enumerate(self.data[state]):
            uid_list.append(sample[-1])
            feat_list.append(sample[0])
            if len(uid_list) == batch_size:
                if self.config['sample_mode'] == 'model':
                    neighbor_list = self.sample_from_model_batch(uid_list, feat_list, sample_num, model)
                elif self.config['sample_mode'] == 'pre_dis_top':
                    neighbor_list = self.sample_from_pre_dis_top(uid_list, sample_num)
                elif self.config['sample_mode'] == 'pre_dis':
                    neighbor_list = self.sample_from_pre_dis(uid_list, sample_num)
                else:
                    raise Exception('wrong sampling mode...')
                for neighbor_set in neighbor_list:
                    # for n in neighbor_set:
                    #     uid_sample_count[n] += 1
                    feat, feat_mask = self.get_neighbor_tensor(neighbor_set)
                    # total_count += feat_mask.size()[0]
                    self.sample_users.append([feat.to(self.device), feat_mask.to(self.device)])
                uid_list, feat_list = [], []
        if len(uid_list) > 0:
            if self.config['sample_mode'] == 'model':
                neighbor_list = self.sample_from_model_batch(uid_list, feat_list, sample_num, model)
            elif self.config['sample_mode'] == 'pre_dis_top':
                neighbor_list = self.sample_from_pre_dis_top(uid_list, sample_num)
            elif self.config['sample_mode'] == 'pre_dis':
                neighbor_list = self.sample_from_pre_dis(uid_list, sample_num)
            else:
                raise Exception('wrong sampling mode...')
            for neighbor_set in neighbor_list:
                # for n in neighbor_set:
                #     uid_sample_count[n] += 1
                feat, feat_mask = self.get_neighbor_tensor(neighbor_set)
                # total_count += feat_mask.size()[0]
                self.sample_users.append([feat.to(self.device), feat_mask.to(self.device)])
        # uid_sample_count = sorted([(uid, count) for uid, count in uid_sample_count.items()])
        # uid_sample_count = map(lambda x: '%s:%s' % (x[0], x[1]), uid_sample_count)
        # print('\t'.join(uid_sample_count))
        # print('%s; state: %s; avg number of sample user: %s' % (get_current_time(), state, total_count / len(self.sample_users)))

    def sample_from_uniform(self, uid, num):
        user_all = set(self.uid_feat_dict.keys())
        user_all.discard(uid)
        sample_users = np.random.choice(list(user_all), num, replace=False)
        return sample_users

    def sample_from_model_batch(self, uid_batch, feat_batch, num, model):
        if self.sample_uid_feat_tensor_dict is None:
            self.generate_users_feat_tensor()
        feat_count = []
        for feat in feat_batch:
            feat_count.append(len(feat))
        max_feat_num = max(feat_count)
        feat_len = feat_batch[0].shape[1]
        user_x = torch.zeros(len(uid_batch), max_feat_num, feat_len, dtype=torch.int32).long().to(self.device)
        user_mask = torch.zeros(len(uid_batch), max_feat_num, dtype=torch.float32).to(self.device)
        for idx, feat in enumerate(feat_batch):
            user_x[idx, :feat_count[idx], :] = feat
            user_mask[idx, :feat_count[idx]] = 1
        uids_list = self.sample_uid_feat_tensor_dict['uids_list']
        other_x = self.sample_uid_feat_tensor_dict['feat_tensor']
        other_mask = self.sample_uid_feat_tensor_dict['feat_mask_tensor']
        scores = model.get_user_sim_scores(user_x, user_mask, other_x, other_mask)
        for idx, uid in enumerate(uid_batch):
            if uid < len(self.sample_uid_feat_tensor_dict['uids_idx']) and self.sample_uid_feat_tensor_dict['uids_idx'][uid] != -1:
                scores[idx][self.sample_uid_feat_tensor_dict['uids_idx'][uid]] = -np.inf
        # topk_idx = scores.argsort(axis=1)[:, -num:]
        # sample_users = uids_list[topk_idx]
        sample_users = []
        for idx in range(len(scores)):
            prob = np.exp(scores[idx] / self.config['sample_temp'])
            prob = prob / np.sum(prob)
            sample_users.append(np.random.choice(uids_list, num, replace=False, p=prob))
        return sample_users

    def generate_users_feat_tensor(self):
        uids_list = np.array(list(self.uid_feat_dict.keys()))
        feat_tensor, feat_mask_tensor = self.get_neighbor_tensor(uids_list)
        max_uid = np.max(uids_list)
        uids_idx = np.ones(max_uid + 1, np.int) * -1
        for idx, uid in enumerate(uids_list):
            uids_idx[uid] = idx
        self.sample_uid_feat_tensor_dict = {'uids_idx': uids_idx,
                                            'uids_list': uids_list,
                                            'feat_tensor': feat_tensor.long().to(self.device),
                                            'feat_mask_tensor': feat_mask_tensor.to(self.device)}

    def sample_from_pre_dis_top(self, uid_list, num):
        sample_users = []
        for uid in uid_list:
            scores = self.implicit_neighbor_matrix[uid]
            scores[uid] = 0
            topk = scores.argsort()[-num:]
            topk = [k for k, score in zip(topk, scores[topk]) if score > 0]
            sample_users.append(topk)
        return sample_users

    def sample_from_pre_dis(self, uid_list, num):
        sample_users = []
        for uid in uid_list:
            scores = self.implicit_neighbor_matrix[uid]
            scores[uid] = 0
            prob = np.log2(scores + 1)
            prob = prob / np.sum(prob)
            sample_users.append(np.random.choice(range(len(prob)), num, replace=False, p=prob))
        return sample_users


if __name__ == '__main__':
    from Config import config_db as config
    from Config import states

    datasets = DataHelper(config, states)

