stages = ['pretrain', 'linear_eval', 'fine_tune']

log_keys = [
    'model_name',
    # 'feature_pooling_method', # useless
]

pretrain = {
        # basic configuration
        'distributed': True,
        'log_path': 'data/log',
        'ckpt_path': 'data/ckpt',
        'ckpt_name': None,
        'batch_sizes': (16, 24, 12),
        'shuffle': (False, False, False),
        'max_epochs': 100,
        'num_workers': 0,
        'log_cmd': True,
        # distributed 
        'fp16_opt_level': "02",
        # dataloader
        'cache_image_vectors': True,
        'datasets': 'imagenet', # 'mnist', 'stl10', 'cifar10', 'cifar100'
        'image_path': 'data/imagenet',
        'image_size': 224,
        'normalize': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],
        # cifar_norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
        'transform': 'sim_siam_transform',
        # model
        'model_name': 'sim_siam',
        'use_inputs': ['x_i', 'x_j'],
        'use_outputs': ['p_i', 'p_j', 'z_i', 'z_j'], # simclr: ['h','z']
        'layers': 3, 
        'dropout': 0.5,
        # loss
        'loss_name': 'sim_siam_loss',
        # optimizer
        'optimizer': 'larc', # lars
        'sub_optimizer': 'sgd' # sgd # adam #adagrad # optimizer to wrap larc
        'learning_rate': 1e-4, # 0.3 x batchsize / 256
        'lr_decay': 0,
        'weight_decay': 1e-6,
        'momentum': 0.9,
        # scheduler
        'scheduler': 'simclr_lr',
        'warm_up': 10,
        'step_size': 0, # for step lr decay
        'gamma': 0.1, # for step lr decay
        'cycles': 0.5 # for cosine lr decay
        'trust_coefficient': 1e-3 # for calculating lr
        'clip': True, # clipping / scaling mode of larc
        'eps': 1e-8, # helps numerical stability while calculating adaptive_lr
        # metric
        'loss_metric': 'none', # mean # sum'
        'metrics': [],
    }

linear_eval = {
        # basic configuration
        'distributed': True,
        'log_path': 'data/log',
        'ckpt_path': 'data/ckpt',
        'ckpt_name': None,
        'batch_sizes': (16, 24, 12),
        'shuffle': (False, False, False),
        'max_epochs': 100,
        'num_workers': 0,
        'log_cmd': True,
        # dataloader
        'cache_image_vectors': True,
        'datasets': 'imagenet', # 'mnist', 'stl10', 'cifar10', 'cifar100'
        'image_path': 'data/imagenet',
        'image_size': 224,
        'normalize': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],
        # cifar_norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
        'transform': 'sim_siam_transform',
        # model
        'model_name': 'sim_siam',
        'use_inputs': ['x_i', 'x_j'],
        'use_outputs': ['p_i', 'p_j', 'z_i', 'z_j'],
        'layers': 3, 
        'dropout': 0.5,
        # loss
        'loss_name': 'sim_siam_loss',
        # optimizer
        'optimizer': 'larc', # lars
        'sub_optimizer': 'sgd' # sgd # adam #adagrad # optimizer to wrap larc
        'learning_rate': 1e-4, # 0.3 x batchsize / 256
        'lr_decay': 0,
        'weight_decay': 1e-6,
        'momentum': 0.9,
        # scheduler
        'scheduler': 'simclr_lr',
        'warm_up': 10,
        'step_size': 0, # for step lr decay
        'gamma': 0.1, # for step lr decay
        'cycles': 0.5 # for cosine lr decay
        'trust_coefficient': 1e-3 # for calculating lr
        'clip': True, # clipping / scaling mode of larc
        'eps': 1e-8, # helps numerical stability while calculating adaptive_lr
        # metric
        'loss_metric': 'none', # mean # sum'
        'metrics': [],
    }

logistic_regression = {
        # basic configuration
        'distributed': True,
        'log_path': 'data/log',
        'ckpt_path': 'data/ckpt',
        'ckpt_name': None,
        'batch_sizes': (16, 24, 12),
        'shuffle': (False, False, False),
        'max_epochs': 100,
        'num_workers': 0,
        'log_cmd': True,
        # dataloader
        'cache_image_vectors': True,
        'datasets': 'imagenet', # 'mnist', 'stl10', 'cifar10', 'cifar100'
        'image_path': 'data/imagenet',
        'image_size': 224,
        'normalize': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],
        # cifar_norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
        'transform': 'sim_siam_transform',
        'num_classes': 10,
        # model
        'model_name': 'logistic_regression',
        'use_inputs': ['x_i', 'x_j'],
        'use_outputs': ['p_i', 'p_j', 'z_i', 'z_j'],
        'layers': 3, 
        'dropout': 0.5,
        # loss
        'loss_name': 'cross_entropy_loss',
        # optimizer
        'sub_optimizer': 'sgd' # sgd # adam #adagrad # optimizer to wrap larc
        'learning_rate': 1e-4, # 0.3 x batchsize / 256
        'lr_decay': 0,
        'weight_decay': 1e-6,
        'momentum': 0.9,
        # scheduler
        'scheduler': 'simclr_lr',
        'warm_up': 10,
        'step_size': 0, # for step lr decay
        'gamma': 0.1, # for step lr decay
        'cycles': 0.5 # for cosine lr decay
        'trust_coefficient': 1e-3 # for calculating lr
        'clip': True, # clipping / scaling mode of larc
        'eps': 1e-8, # helps numerical stability while calculating adaptive_lr
        # metric
        'loss_metric': 'none', # mean # sum'
        'metrics': [],
    }

fine_tune = {
        # basic configuration
        'distributed': True,
        'log_path': 'data/log',
        'ckpt_path': 'data/ckpt',
        'ckpt_name': None,
        'batch_sizes': (16, 24, 12),
        'shuffle': (False, False, False),
        'max_epochs': 100,
        'num_workers': 0,
        'log_cmd': True,
        # dataloader
        'cache_image_vectors': True,
        'datasets': 'imagenet', # 'mnist', 'stl10', 'cifar10', 'cifar100'
        'image_path': 'data/imagenet',
        'image_size': 224,
        'normalize': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],
        # cifar_norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
        'transform': 'sim_siam_transform',
        # model
        'model_name': 'fine_tuning_model',
        'use_inputs': ['x_i', 'x_j'],
        'use_outputs': ['p_i', 'p_j', 'z_i', 'z_j'],
        'layers': 3, 
        'dropout': 0.5,
        # loss
        'loss_name': 'cross_entropy_loss',
        # optimizer
        'sub_optimizer': 'sgd' # sgd # adam #adagrad # optimizer to wrap larc
        'learning_rate': 1e-4, # 0.3 x batchsize / 256
        'lr_decay': 0,
        'weight_decay': 1e-6,
        'momentum': 0.9,
        'nesterov': True,
        # scheduler
        'scheduler': 'step_lr',
        'warm_up': 10,
        'step_size': 0, # for step lr decay
        'gamma': 0.1, # for step lr decay
        'cycles': 0.5 # for cosine lr decay
        'trust_coefficient': 1e-3 # for calculating lr
        'clip': True, # clipping / scaling mode of larc
        'eps': 1e-8, # helps numerical stability while calculating adaptive_lr
        # metric
        'loss_metric': 'none', # mean # sum'
        'metrics': [],
    }
