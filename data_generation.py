import os
import json
import torch
import numpy as np
import random
import pickle
import collections
from tqdm import tqdm

from Config import states
from dataset import dbook, yelp

random.seed(13)


def create_item_user_map(dataset, user2id, item2id, user_in_meta_training):
    uid_iids_map = dict()
    iid_uids_map = collections.defaultdict(set)
    for user, items in dataset.items():
        if int(user) not in user_in_meta_training:
            continue
        uid = user2id[int(user)]
        uid_iids_map[uid] = [item2id[item] for item in items]
        for item in items:
            iid = item2id[item]
            iid_uids_map[iid].add(uid)
    return uid_iids_map, iid_uids_map


def get_coclick_neighbor(iids, uid_iids_map, iid_uids_map, max_num):
    iids = list(iids.numpy())
    iids = set(iids)
    candidate_uids = set()
    coclick_neighbor_list = list()
    for iid in iids:
        candidate_uids.update(iid_uids_map[iid])
    for uid in candidate_uids:
        nume = iids.intersection(uid_iids_map[uid])
        similarity = len(nume) / len(iids)
        coclick_neighbor_list.append((similarity, uid))
    coclick_neighbor_list.sort(key=lambda x: -x[0])
    coclick_neighbor_list = [uid for sim, uid in coclick_neighbor_list[:max_num]]
    return coclick_neighbor_list


def find_other_uids(uid, iid, iid_uid_dict):
    uid_prob = collections.defaultdict(float)
    uid_set = set(iid_uid_dict[iid])
    uid_set.discard(uid)
    if len(uid_set) == 0:
        return uid_prob
    prob = 1. / len(uid_set)
    for uid in uid_set:
        uid_prob[uid] = prob
    return uid_prob


def find_other_iids(iid, iid_fid_dict, fid_iid_dict):
    iid_prob = collections.defaultdict(float)
    iid_fid_prob = 1. / len(iid_fid_dict[iid])
    for fid in iid_fid_dict[iid]:
        iid_set = fid_iid_dict[fid]
        iid_set.discard(iid)
        if len(iid_set) == 0:
            continue
        fid_iid_prob = 1. / len(iid_set)
        for iid in iid_set:
            iid_prob[iid] += iid_fid_prob * fid_iid_prob
    return iid_prob


def get_implicit_neighbor_for_single(uid, rate_uid_iid_dict, rate_iid_uid_dict, feat_iid_fid_dict, feat_fid_iids_dict):
    """
    根据uid找到可能的邻居
    :param uid:
    :param rate_uid_iid_dict:
    :param rate_iid_uid_dict:
    :param feat_iid_fid_dict:
    :param feat_fid_iids_dict:
    :return:
    """
    suid_prob = collections.defaultdict(float)
    for rate, uid_iid_dict in rate_uid_iid_dict.items():
        if uid not in uid_iid_dict:
            continue
        prob_uid_iid = 1. / len(uid_iid_dict[uid])
        for iid in uid_iid_dict[uid]:
            uid_prob = find_other_uids(uid, iid, rate_iid_uid_dict[rate])
            for k, v in uid_prob.items():
                suid_prob[k] += prob_uid_iid * v
            for feat_name in feat_iid_fid_dict.keys():
                s_iid_prob = find_other_iids(iid, feat_iid_fid_dict[feat_name], feat_fid_iids_dict[feat_name])
                for s_iid, s_prob in s_iid_prob.items():
                    uid_prob = find_other_uids(uid, s_iid, rate_iid_uid_dict[rate])
                    for k, v in uid_prob.items():
                        suid_prob[k] += prob_uid_iid * s_prob * v
    imp_neighbor_score = [(k, v) for k, v in suid_prob.items()]
    imp_neighbor_score.sort(key=lambda x: -x[1])
    return imp_neighbor_score


def get_heterogeneous_graph(dataset_path, item_feat_list, rate_list):
    """
    构建异构图
    :param dataset_path:
    :param item_feat_list: item使用的特征
    :param rate_list: 所有可能的打分
    :return:
    """
    data = pickle.load(open("{}/item_feature.pkl".format(dataset_path), "rb"))
    item_feat_name_list = data[0]
    print('create implicit neighbor matrix...')
    feature_info = dict()
    for feat in item_feat_list:
        feature_info[feat] = 0
    for feat_idx, feature in enumerate(item_feat_name_list):
        if feature in feature_info:
            feature_info[feature] = feat_idx
    iid2feats_dict = dict()
    feat2iids_dict = dict()
    for feat in item_feat_list:
        iid2feats_dict[feat] = collections.defaultdict(set)
        feat2iids_dict[feat] = collections.defaultdict(set)
    rate_uid_iid_dict = dict()
    rate_iid_uid_dict = dict()
    for rate in rate_list:
        rate_uid_iid_dict[rate] = collections.defaultdict(set)
        rate_iid_uid_dict[rate] = collections.defaultdict(set)

    for state in states:
        print('load info from %s' % state)
        support_x_list = pickle.load(open("{}/{}/supp_x.pkl".format(dataset_path, state), "rb"))
        support_y_list = pickle.load(open("{}/{}/supp_y.pkl".format(dataset_path, state), "rb"))
        for sup_x, sup_y in zip(support_x_list, support_y_list):
            uid, iids, feats_list = sup_x['uid'], sup_x['iid_arr'], sup_x['feat_list']
            for iid, feats, label in zip(iids, feats_list, sup_y):
                label = int(label)
                iid = int(iid)
                rate_uid_iid_dict[label][uid].add(iid)
                rate_iid_uid_dict[label][iid].add(uid)
                for feat_name, feat_idx in feature_info.items():
                    iid2feats_dict[feat_name][iid].add(int(feats[feat_idx]))
                    feat2iids_dict[feat_name][int(feats[feat_idx])].add(iid)
    return rate_iid_uid_dict, rate_uid_iid_dict, iid2feats_dict, feat2iids_dict


def get_index_mapping(dataset, output_dir):
    """
    获得user、item和对应feature的index映射
    :param dataset:
    :param output_dir:
    :return:
    """
    user2id = dict()
    item2id = dict()
    userFeat2id = dict()
    itemFeat2id = dict()
    user_set = set(dataset.rating_data['user']).intersection(set(dataset.user_feature['user']))
    for user in user_set:
        user2id[user] = len(user2id)
    item_set = set(dataset.rating_data['item']).intersection(set(dataset.item_feature['item']))
    for item in item_set:
        item2id[item] = len(item2id)
    user_feat_name_list = list(dataset.user_feature.drop('user', axis=1).keys())
    for feat_name in user_feat_name_list:
        userFeat2id[feat_name] = dict()
        features = dataset.user_feature[feat_name].unique()
        for feat in features:
            userFeat2id[feat_name][feat] = len(userFeat2id[feat_name])
    item_feat_name_list = list(dataset.item_feature.drop('item', axis=1).keys())
    for feat_name in item_feat_name_list:
        itemFeat2id[feat_name] = dict()
        features = dataset.item_feature[feat_name].unique()
        for feat in features:
            itemFeat2id[feat_name][feat] = len(itemFeat2id[feat_name])
    print("num_user: %s" % len(user2id))
    print("num_item: %s" % len(item2id))
    for feat_name, items in userFeat2id.items():
        print("num_%s: %s" % (feat_name, len(items)))
    for feat_name, items in itemFeat2id.items():
        print("num_%s: %s" % (feat_name, len(items)))
    pickle.dump(user2id, open("{}/user2id.pkl".format(output_dir), "wb"))
    pickle.dump(item2id, open("{}/item2id.pkl".format(output_dir), "wb"))
    pickle.dump(userFeat2id, open("{}/userFeat2id.pkl".format(output_dir), "wb"))
    pickle.dump(itemFeat2id, open("{}/itemFeat2id.pkl".format(output_dir), "wb"))
    return user2id, item2id, userFeat2id, itemFeat2id, user_set, item_set, user_feat_name_list, item_feat_name_list


def transfer_feat_to_index(dataset, user_set, user_feat_name_list, userFeat2id, item_set, item_feat_name_list, itemFeat2id, output_dir):
    """
    把user和item的feature转化成index
    :param dataset:
    :param user_set:
    :param user_feat_name_list:
    :param userFeat2id:
    :param item_set:
    :param item_feat_name_list:
    :param itemFeat2id:
    :param output_dir:
    :return:
    """
    item_feature = dict()
    for item in item_set:
        features = dataset.item_feature[dataset.item_feature['item'] == item]
        feat_id_list = []
        for feat_name in item_feat_name_list:
            feat = list(features[feat_name])[0]
            feat_id_list.append(itemFeat2id[feat_name][feat])
        item_feature[item] = feat_id_list
    print('item feature name in order: ', '\t'.join(item_feat_name_list))
    pickle.dump([item_feat_name_list, item_feature], open("{}/item_feature.pkl".format(output_dir), "wb"))
    user_feature = dict()
    for user in user_set:
        features = dataset.user_feature[dataset.user_feature['user'] == user]
        feat_id_list = []
        for feat_name in user_feat_name_list:
            feat = list(features[feat_name])[0]
            feat_id_list.append(userFeat2id[feat_name][feat])
        user_feature[user] = feat_id_list
    print('user feature name in order: ', '\t'.join(user_feat_name_list))
    pickle.dump([user_feat_name_list, user_feature], open("{}/user_feature.pkl".format(output_dir), "wb"))
    return item_feature, user_feature


def transfer_neighbor_to_index(dataset, user_set, user2id, user_in_meta_training, output_dir):
    """
    把邻居的id转化成index
    :param dataset:
    :param user_set:
    :param user2id:
    :param user_in_meta_training:
    :param output_dir:
    :return:
    """
    uidNeigh2id = dict()
    for user in user_set:
        uid = user2id[user]
        if user not in dataset.user_neighbor:
            uidNeigh2id[uid] = []
            continue
        neighbor_list = dataset.user_neighbor[user]
        neigh2id_list = [user2id[neighbor] for neighbor in set(neighbor_list) if
                         neighbor in user_set and neighbor in user_in_meta_training]
        uidNeigh2id[uid] = neigh2id_list
    pickle.dump(uidNeigh2id, open("{}/user_neighbor.pkl".format(output_dir), "wb"))
    return uidNeigh2id


def get_task_tensor(uid_list, feats_list, labels_list, all_label_list):
    if len(uid_list) == 0:
        return torch.tensor([]), torch.tensor([])
    uid_rate_feat_dict = dict()
    for idx, uid in enumerate(uid_list):
        uid_rate_feat_dict[uid] = dict()
        feats, labels = feats_list[idx], labels_list[idx]
        for feat, label in zip(feats, labels):
            label = int(label)
            if label not in uid_rate_feat_dict[uid]:
                uid_rate_feat_dict[uid][label] = []
            uid_rate_feat_dict[uid][label].append(feat)
    max_feat_count = 0
    for uid in uid_list:
        for label in uid_rate_feat_dict[uid].keys():
            if len(uid_rate_feat_dict[uid][label]) > max_feat_count:
                max_feat_count = len(uid_rate_feat_dict[uid][label])
    feat_len = len(feats_list[0][0])
    feat_tensor = torch.zeros(len(all_label_list), len(uid_list), max_feat_count, feat_len, dtype=torch.int32)
    feat_mask_tensor = torch.zeros(len(all_label_list), len(uid_list), max_feat_count, dtype=torch.float32)
    for idx_u, uid in enumerate(uid_list):
        for idx_l, label in enumerate(all_label_list):
            if label not in uid_rate_feat_dict[uid]:
                continue
            feat_count = len(uid_rate_feat_dict[uid][label])
            feat_tensor[idx_l, idx_u, :feat_count, :] = torch.stack(uid_rate_feat_dict[uid][label], dim=0)
            feat_mask_tensor[idx_l, idx_u, :feat_count] = 1
    return feat_tensor, feat_mask_tensor


def get_training_samples(dataset_path, user2id, item2id, user_feature, item_feature, output_dir, **kwargs):
    """
    构建训练和测试用的数据
    :param dataset_path:
    :param user2id:
    :param item2id:
    :param user_feature:
    :param item_feature:
    :param output_dir:
    :param kwargs:
    :return:
    """
    item_num = []
    user_num = []
    rating_num = 0
    for state in states:
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []
        for _, (user, items) in tqdm(enumerate(dataset.items())):
            labels = dataset_y[user]
            user = int(user)
            uid = user2id[user]
            assert len(items) == len(labels)
            item_iid_label_ids = [(item, item2id[item], label) for item, label in zip(items, labels)]
            item_len = len(item_iid_label_ids)

            if item_len < 13 or item_len > 23:
                continue
            rating_num = rating_num + item_len

            random.shuffle(item_iid_label_ids)
            item_arr = np.array([item for item, iid, label in item_iid_label_ids])
            iid_arr = np.array([iid for item, iid, label in item_iid_label_ids])
            labels_arr = np.array([label for item, iid, label in item_iid_label_ids])
            for item in item_arr:
                if item not in item_num:
                    item_num.append(item)
            if user not in user_num:
                user_num.append(user)
            featid_list = []
            for item in item_arr[:-10]:
                featid_list.append(item_feature[item] + user_feature[user])
            sample = {'uid': uid,
                      'iid_arr': torch.tensor(iid_arr[:-10], dtype=torch.int32),
                      'feat_list': torch.tensor(featid_list, dtype=torch.int32)}
            support_x_list.append(sample)
            support_y_list.append(torch.tensor(labels_arr[:-10], dtype=torch.int32))
            featid_list = []
            for item in item_arr[-10:]:
                featid_list.append(item_feature[item] + user_feature[user])
            sample = {'uid': uid,
                      'iid_arr': torch.tensor(iid_arr[-10:], dtype=torch.int32),
                      'feat_list': torch.tensor(featid_list, dtype=torch.int32)}
            query_x_list.append(sample)
            query_y_list.append(torch.tensor(labels_arr[-10:], dtype=torch.int32))

        if not os.path.exists("{}/{}".format(output_dir, state)):
            os.mkdir("{}/{}".format(output_dir, state))
        pickle.dump(support_x_list, open("{}/{}/supp_x.pkl".format(output_dir, state), "wb"))
        pickle.dump(support_y_list, open("{}/{}/supp_y.pkl".format(output_dir, state), "wb"))
        pickle.dump(query_x_list, open("{}/{}/query_x.pkl".format(output_dir, state), "wb"))
        pickle.dump(query_y_list, open("{}/{}/query_y.pkl".format(output_dir, state), "wb"))
    print(len(item_num))
    print(rating_num)
    print(len(user_num))
    meta_training_feat_dict = dict()
    support_x_list = pickle.load(open("{}/{}/supp_x.pkl".format(output_dir, 'meta_training'), "rb"))
    support_y_list = pickle.load(open("{}/{}/supp_y.pkl".format(output_dir, 'meta_training'), "rb"))
    for sup_x, sup_y in zip(support_x_list, support_y_list):
        uid, feat_list = sup_x['uid'], sup_x['feat_list']
        meta_training_feat_dict[uid] = [feat_list, sup_y]

    uid_iids_map, iid_uids_map = kwargs['uid_iids_map'], kwargs['iid_uids_map']
    uidNeigh2id = kwargs['uidNeigh2id']
    all_label_list = kwargs['rating_list']
    item_feat_list_for_graph = kwargs['item_feat_list_for_graph']
    rate_iid_uid_dict, rate_uid_iid_dict, feat_iid_fid_dict, feat_fid_iid_dict = get_heterogeneous_graph(output_dir, item_feat_list_for_graph, all_label_list)
    pickle.dump(all_label_list, open("{}/all_label_in_order.pkl".format(output_dir), "wb"))
    for state in states:
        support_x_list = pickle.load(open("{}/{}/supp_x.pkl".format(output_dir, state), "rb"))
        support_y_list = pickle.load(open("{}/{}/supp_y.pkl".format(output_dir, state), "rb"))
        social_info_list = []
        for _, (sup_x, sup_y) in tqdm(enumerate(zip(support_x_list, support_y_list))):
            # 自己的特征tensor
            uid, iid_arr, feat_list = sup_x['uid'], sup_x['iid_arr'], sup_x['feat_list']
            self_feat_tensor, self_feat_mask_tensor = get_task_tensor([uid], [feat_list], [sup_y], all_label_list)

            # 社交网络上邻居的特征tensor
            neighbor_list = uidNeigh2id[uid]
            neighbor_feat_list, neighbor_label_list = [], []
            for neighbor in neighbor_list:
                neighbor_feat_list.append(meta_training_feat_dict[neighbor][0])
                neighbor_label_list.append(meta_training_feat_dict[neighbor][1])
            neighbor_feat_tensor, neighbor_feat_mask_tensor = get_task_tensor(neighbor_list, neighbor_feat_list, neighbor_label_list, all_label_list)

            # 协同过滤得到的邻居特征tensor
            coclick_list = get_coclick_neighbor(iid_arr, uid_iids_map, iid_uids_map, kwargs['implicit_num'])
            coclick_feat_list, coclick_label_list = [], []
            for coclick in coclick_list:
                coclick_feat_list.append(meta_training_feat_dict[coclick][0])
                coclick_label_list.append(meta_training_feat_dict[coclick][1])
            coclick_feat_tensor, coclick_feat_mask_tensor = get_task_tensor(coclick_list, coclick_feat_list, coclick_label_list, all_label_list)

            # 隐藏邻居的特征tensor
            imp_neighbor = get_implicit_neighbor_for_single(uid, rate_uid_iid_dict, rate_iid_uid_dict, feat_iid_fid_dict, feat_fid_iid_dict)
            idx = 0
            implicit_list = []
            implicit_feat_list, implicit_label_list = [], []
            while len(implicit_list) < kwargs['implicit_num'] and idx < len(imp_neighbor):
                neighbor = imp_neighbor[idx][0]
                if neighbor in meta_training_feat_dict:
                    implicit_list.append(neighbor)
                    implicit_feat_list.append(meta_training_feat_dict[neighbor][0])
                    implicit_label_list.append(meta_training_feat_dict[neighbor][1])
                idx += 1
            implicit_feat_tensor, implicit_feat_mask_tensor = get_task_tensor(implicit_list, implicit_feat_list, implicit_label_list, all_label_list)

            sample = {'self_feat': self_feat_tensor,
                      'self_mask': self_feat_mask_tensor,
                      'social_feat': neighbor_feat_tensor,
                      'social_mask': neighbor_feat_mask_tensor,
                      'coclick_feat': coclick_feat_tensor,
                      'coclick_mask': coclick_feat_mask_tensor,
                      'implicit_feat': implicit_feat_tensor,
                      'implicit_mask': implicit_feat_mask_tensor}
            social_info_list.append(sample)
        pickle.dump(social_info_list, open("{}/{}/social_info.pkl".format(output_dir, state), "wb"))


def process(data_set, item_feat_list, implicit_num):
    """
    生成数据
    :param data_set:
    :param item_feat_list: 生成异构图用的特征
    :param implicit_num: 隐藏好友的数量
    :return:
    """
    dataset_path = "data/"
    output_dir = "data_process/"
    if data_set == 'dbook':
        dataset_path += 'dbook'
        output_dir += 'dbook-20'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        dataset = dbook()
    elif data_set == 'yelp':
        dataset_path += 'yelp'
        output_dir += 'yelp'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        dataset = yelp()
    else:
        raise Exception('wrong dataset name')

    # construct mapping between user, item, features and index
    user2id, item2id, userFeat2id, itemFeat2id, user_set, item_set, user_feat_name_list, item_feat_name_list = get_index_mapping(dataset, output_dir)

    user_in_meta_training = set()
    with open("{}/meta_training.json".format(dataset_path), encoding="utf-8") as f:
        meta_training_data = json.loads(f.read())
        for user, items in meta_training_data.items():
            if len(items) < 13 or len(items) > 100:
                continue
            user_in_meta_training.add(int(user))

    # change user neighbor, user feature, item feature to index
    uidNeigh2id = transfer_neighbor_to_index(dataset, user_set, user2id, user_in_meta_training, output_dir)
    features = transfer_feat_to_index(dataset, user_set, user_feat_name_list, userFeat2id, item_set, item_feat_name_list, itemFeat2id, output_dir)
    item_feature, user_feature = features

    # add co-click neighbors
    with open("{}/{}.json".format(dataset_path, 'meta_training'), encoding="utf-8") as f:
        meta_training = json.loads(f.read())
    uid_iids_map, iid_uids_map = create_item_user_map(meta_training, user2id, item2id, user_in_meta_training)

    # create training samples
    other_data = {'uid_iids_map': uid_iids_map,
                  'iid_uids_map': iid_uids_map,
                  'uidNeigh2id': uidNeigh2id,
                  'rating_list': dataset.rating_list,
                  'item_feat_list_for_graph': item_feat_list,
                  'implicit_num': implicit_num}
    get_training_samples(dataset_path, user2id, item2id, user_feature, item_feature, output_dir, **other_data)


def neighbor_statistic():
    output_dir = "data_process/dbook_with_coclick"
    valid_user_set = set()
    for state in states:
        neighbor_cnt_dict = collections.defaultdict(int)
        coclick_neighbor_cnt_dict = collections.defaultdict(int)
        support_x_list = pickle.load(open("{}/{}/supp_x.pkl".format(output_dir, state), "rb"))
        if state == 'meta_training':
            for item in support_x_list:
                valid_user_set.add(item['uid'])
        for item in support_x_list:
            neighbor_list = [uid for uid in item['neighbor_list'].numpy()]
            neighbor_cnt_dict[len(neighbor_list)] += 1
            coclick_list = [uid for uid in item['coclick_list'].numpy() if uid in valid_user_set]
            coclick_neighbor_cnt_dict[len(coclick_list)] += 1
        neighbor_cnt_list = [(length, count) for length, count in neighbor_cnt_dict.items()]
        neighbor_cnt_list.sort()
        print(state)
        total = 0
        user_count = 0
        for length, count in neighbor_cnt_list:
            print(length, count)
            total += length * count
            user_count += count
        print(total / user_count)
        coclick_neighbor_cnt_list = [(length, count) for length, count in coclick_neighbor_cnt_dict.items()]
        coclick_neighbor_cnt_list.sort()
        total = 0
        user_count = 0
        for length, count in coclick_neighbor_cnt_list:
            print(length, count)
            total += length * count
            user_count += count
        print(total / user_count)


if __name__ == "__main__":
    data_set = 'dbook'
    item_feat_list = ['publisher']
    process(data_set, item_feat_list, 50)

    # neighbor_statistic()
