config = {
    'extractor_batch_size': 32, 
    'model_name': 'simclr', 
    'log_path': 'data/log',
    'tokenizer': 'nonword', # 'nltk' # need to check
    'batch_sizes':  (16, 24, 12),
    'lower': True,
    'use_inputs':['images',ker'],
    'stream_type': ['script',  'visual_bb', 'visual_meta'], #
    'cache_image_vectors': True,
    'image_path': 'data/images',
    'visual_path': 'data/AnotherMissOh/AnotherMissOh_Visual.json',
    'data_path': 'dataAnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_set_script.jsonl',
    'subtitle_path': 'data/AnotherMissOh/AnotherMissOh_script.json',
    'glove_path': "data/glove.6B.300d.txt", # "data/glove.6B.50d.txt"
    'vocab_path': "data/AnotherMissOh/vocab.pickle",
    'val_type': 'all', #  'all' | 'ch_only'
    'max_epochs': 20,
    'num_workers': 40, 
    'image_dim': 512,  # hardcoded for ResNet18
    'n_dim': 300,  
    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'loss_name': 'cross_entropy_loss',
    'optimizer': 'adam',
    'metrics': [],
    'log_cmd': False,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'max_sentence_len': 30,
    'max_sub_len': 300,
    'max_image_len': 100,
    'shuffle': (False, False, False)
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    # 'feature_pooling_method', # useless
]/
