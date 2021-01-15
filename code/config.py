config = { 
    'model_name': 'sim_siam', 
    'log_path': 'data/log',
    'batch_sizes':  (16, 24, 12),
    'use_inputs':['x_i', 'x_j'],
    'use_outputs': ['p_i', 'p_j', 'z_i', 'z_j'],
    'loss_metric': 'none' # mean # sum
    'stream_type': ['visual'], #
    'cache_image_vectors': True,
    'datasets': 'imagenet',
    'image_path': 'data/imagenet', #data/cifar10 #data/cifar100
    'val_type': 'all', #  'all' | 'ch_only'
    'max_epochs': 100,
    'num_workers': 0, 
    'image_dim': 512,  # hardcoded for ResNet18
    'n_dim': 300,  
    'layers': 3,
    'dropout': 0.5,
    'optimizer': 'larc', # lars
    'sub_optimizer': 'sgd', # optimizer to wrap lars # sgd # adam # adagrad
    'learning_rate': 1e-4, # 0.3 x BatchSize / 256
    'lr_decay': 0, 
    'weight_decay': 1e-6,
    'momentum': 0.9,
    'scheduler': 'simclrlr',
    'trust_coefficient': 1e-3, # trust coefficient for calculating lr. 
    'clip': True # clipping/scaling mode of LARC
    'eps': 1e-8, # epilog klunge to help with numerical stability while calculating adaptive_lr
    'warm_up': 10, # learning rate schedule
    'step_size': 0, # for step lr decay
    'gamma': 0.1, # for step lr decay
    'cycles': 0.5, # for cosine lr decay
#    'min_lr': 1e-4,
#    'last_epoch': -1,
    'loss_name': 'sim_siam_loss',
    'metrics': [],
    'log_cmd': True,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'shuffle': (False, False, False)
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    # 'feature_pooling_method', # useless
]
