'''
Semi-supervised learning:
sample 1% or 10% of the labeled ILSVRC-12 training datasets in a class-balanced way.
Simply fine-tune the whole base network on the labeled data without regularization.
Interestingly, fine-tuning our pretrained ResNet-50 on full ImageNet are also significantly bettter then training from scratch.
'''

'''
Semi-supervised learning via Fine-tuning:
Fine-tune using the Nesterov momentum optimizer with a batch size of 4096, momentum of 0.9, and learning rate of 0.8 without warmup.
Only random cropping is used for preprocesssing.
For 1% labeled data we fine-tune for 60 epochs, for 10% labeled data we fine-tune for 30 epochs.
'''
'''
Transfer via Fine-Tuning:
Fine-tune the entire network using the weights of the prtrained network as initialization.
'''
