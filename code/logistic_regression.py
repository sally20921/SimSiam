'''
transfer learning:
1. linear evaluation: a logistic regression classifier is trained to classify a new dataset based on self-supervised representation learned on ImageNet.
2. fine-tuning: where we allow all weights to vary during training.
'''

'''
Transfer via a Linear Classifier:
Train an l2-regularized multinomial logistic regression classifier on features extracted from the frozen pretrained network.
'''
