from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch
import gc
import glob
import os
import random
import sys
import time
import numpy as np
import torch
import json
from SMR import SMR
from DataHelper import DataHelper
from tqdm import tqdm
from Config import states
from util import *

random.seed(13)
np.random.seed(13)
torch.manual_seed(13)


def training(model, model_save, model_file_path, data):
    output_to_file('%s: training model...' % get_current_time(), log_file)
    if config['use_cuda']:
        model.cuda()
    model.train()

    batch_size = config['batch_size']
    num_epoch = config['num_epoch']
    train_data = data.data['meta_training']
    max_metrics = dict()
    metrics_name = ['mae', 'rmse', 'ndcg@5']
    for state in states:
        if state != 'meta_training':
            max_metrics[state] = [100., 100., 0]

    for epoch in range(num_epoch):
        loss = []
        start = time.time()
        random.shuffle(train_data)
        num_batch = int(len(train_data) / batch_size)
        # model.reset_time()
        for k in range(num_batch):  # each batch contains some tasks (each task contains a support set and a query set)
            batch_data = data.get_batch('meta_training', k)
            global_step = epoch * num_batch + k
            _loss = model.global_update(global_step, batch_data)
            loss.append(_loss)
        # model.print_time()
        output_to_file('{}: epoch: {}, loss: {:.6f}, cost time: {:.1f}s'.
                       format(get_current_time(), epoch, np.mean(loss), time.time() - start), log_file)
        model.writer.add_scalar('epoch_train_loss', np.mean(loss), global_step=epoch)
        #if epoch < 80 and epoch % 10 != 0:
            #continue
        metrics_update = testing(epoch, model, data, max_metrics)
        model.train()

        if model_save and metrics_update:
            for i in range(3):
                max_ndcg_list = ["%s:%.4f" % (state, metrics[i]) for state, metrics in max_metrics.items()]
                output_to_file('\033[0;31m' + metrics_name[i] + '\t' + '\t'.join(max_ndcg_list) + '\033[0m', log_file)
            output_to_file('saving model...', log_file)
            model_file = os.path.join(model_file_path, 'model_%s' % epoch)
            torch.save(model.state_dict(), model_file)


def testing(epoch, model, data, metrics_dict):
    if config['use_cuda']:
        model.cuda()
    model.eval()
    update = False
    for state in states:
        if state == 'meta_training':
            continue
        loss, mae, rmse, ndcg_at_5 = [], [], [], []
        test_data = data.get_batch(state, -1, True)
        for j in range(len(test_data['supp_xs'])):  # each task
            task_data = {'supp_x': test_data['supp_xs'][j],
                         'supp_y': test_data['supp_ys'][j],
                         'query_x': test_data['query_xs'][j],
                         'query_y': test_data['query_ys'][j],
                         'task_self': test_data['task_self_s'][j],
                         'task_social': test_data['task_social_s'][j],
                         'task_implicit': test_data['task_implicit_s'][j],
                         'task_coclick': test_data['task_coclick_s'][j]}
            _loss, _mae, _rmse, _ndcg_5 = model.evaluation(task_data)
            loss.append(_loss)
            mae.append(_mae)
            rmse.append(_rmse)
            ndcg_at_5.append(_ndcg_5)
        loss_ = torch.stack(loss).mean(0)
        loss_ = loss_.cpu().data.numpy()
        mae_mean = np.mean(mae)
        rmse_mean = np.mean(rmse)
        ndcg_mean = np.mean(ndcg_at_5)
        model.writer.add_scalar('epoch_test_%s_loss' % state, loss_, global_step=epoch)
        model.writer.add_scalar('epoch_test_%s_mae' % state, mae_mean, global_step=epoch)
        model.writer.add_scalar('epoch_test_%s_rmse' % state, rmse_mean, global_step=epoch)
        model.writer.add_scalar('epoch_test_%s_ndcg_at_5' % state, ndcg_mean, global_step=epoch)
        output_to_file('{}: state: {}; loss: {:.5f},mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
                       format(get_current_time(), state, loss_, mae_mean, rmse_mean, ndcg_mean), log_file)
        if mae_mean < metrics_dict[state][0]:
            metrics_dict[state][0] = mae_mean
            update = True
        if rmse_mean < metrics_dict[state][1]:
            metrics_dict[state][1] = rmse_mean
            update = True
        if ndcg_mean > metrics_dict[state][2]:
            metrics_dict[state][2] = ndcg_mean
            update = True
    return update


if __name__ == "__main__":
    data_set = 'dbook'
    # data_set = 'yelp'

    load_model = True

    if data_set == 'dbook':
        from Config import config_db as config
    elif data_set == 'yelp':
        from Config import config_yelp as config
    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])
    # else:
    #     os.system('rm -r %s' % (config['output_dir'] + '/*'))
    with open(os.path.join(config['output_dir'], 'param.json'), 'w') as fout:
        output_to_file(json.dumps(config), fout)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

    model_filepath = os.path.join(config['output_dir'], 'model')
    if not os.path.exists(model_filepath):
        os.mkdir(model_filepath)
    log_file = open(os.path.join(config['output_dir'], 'train_log.txt'), 'w')
    # load data
    output_to_file('%s: loading data...' % get_current_time(), log_file)
    datasets = DataHelper(config, states)

    # training model.
    model = SMR(config)

    if not load_model:
        training(model, model_save=True, model_file_path=model_filepath, data=datasets)
    else:
        test_epoch = 0
        model_file_name = os.path.join(model_filepath, 'model_%s' % test_epoch)
        trained_state_dict = torch.load(model_file_name)
        model.load_state_dict(trained_state_dict)

    # testing
    # testing(hml, device=cuda_or_cpu)
    # print('--------------- {} ---------------'.format(model_name))
