config_db = {
    'input_dir': 'data_process/dbook',
    'output_dir': 'res/dbook',
    'dataset': 'dbook',
    'use_cuda': True,
    'gpu': '2',

    # user
    'num_location': 453,
    'num_fea_user': 1,
    'use_fea_user': 1,

    # item
    'num_publisher': 1789,
    'num_author': 10713,
    'num_fea_item': 2,
    'use_fea_item': 1,

    'embedding_dim': 32,

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    'dropout': 0.2,

    'local_update': 1,
    'lr': 1e-3,
    'local_lr': 1e-3,
    'weight_decay': 3e-3,
    'batch_size': 64,  # for each batch, the number of tasks
    'num_epoch': 120,

    # option
    'social_num': 5,
    'implicit_num': 5,
    'use_coclick': False,# Based on CF, useless, do not change  
    # 'coclick_num': 20,
}

states = ["meta_training", "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]

