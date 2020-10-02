# -*- coding: utf-8 -*-
"""
Created on Fri May 1 00:00:00 2020

@author: Yann
"""
import sys
sys.path.insert(1,"src")
from misc_funcs import RANDOM_SEED,EXPLORE_K,BIG_K,NB_EPOCHS,sep
import torch
import dlc_practical_prologue as dl
import torch.nn as nn
from torch import optim
from time import sleep



from opt import (Net2,train_model,get_accuracy,Kfold_CV,
    Naive,WeightAux,get_double_accuracy,train_double_model,Kfold_CVdouble,)
torch.manual_seed(13)

N_SAMPLES = 500
a, b, c, d = dl.load_data(flatten=False)
train2_input, train2_target, train2_classes = dl.mnist_to_pairs(N_SAMPLES, a, b)
test2_input, test2_target, test2_classes = dl.mnist_to_pairs(N_SAMPLES, c, d)

print("\n\nWelcome to test.py! \n\nThis is where the best parameters of this project are displayed! \n2 runs are gonna be carried out: \nThe first consists in a baseline training 2 parallel CNN to recognize digits and performs a simple comparison (cf report/source code/explore.ipynb) \nThe second consists in a weight sharing and auxiliary loss model\n")
print("Both models will first execute a 12 fold CV in order to obtain robust estimates of their respective accuracies. A classic train/test run is executed as well in order to really make sure that no overfits occures.\n\n")
sleep(1)
# Naive Model
############################
# Parameters definition
crit_comp = nn.CrossEntropyLoss
crit_class = nn.L1Loss
optimizer = optim.AdamW
lr = 1e-2
naive_baseline = Naive()
############################
K = 10
print("#"*20 + " Naive Model " + "#"*20)
print("Computing {} Fold CV...".format(K))
_ = Kfold_CVdouble(train2_input,train2_target,train2_classes,
                   Naive(),crit_comp,crit_class,optimizer,lr,K=K,verbose=True)
print(sep)
print("Training on train set...")
train_double_model(train2_input,train2_target,train2_classes,naive_baseline,crit_comp,crit_class,optimizer,lr,nb_epochs=25,verbose=True)
print(sep + "Test" + sep)
_ = get_double_accuracy(naive_baseline, test2_input, test2_target, test2_classes,verbose=True)


# Weight sharing model
############################
crit_comp = nn.L1Loss
crit_class = nn.CrossEntropyLoss
optimizer = optim.Adam
lr = 1e-2
lambda_ = 0.2
weight_share_net = WeightAux(True, True)
############################
print("\n")
print("#"*20 + " Weight Sharing/Aux loss " + "#"*20)
print("Computing {} Fold CV...".format(K))
acc_list = Kfold_CVdouble(train2_input,train2_target,train2_classes,WeightAux(True, True),crit_comp,crit_class,
                          optimizer,lr,lambda_,K=K,verbose=True)
print("Training on train set...")

train_double_model(train2_input,train2_target,train2_classes,weight_share_net,crit_comp,crit_class,optimizer,lr,lambda_,nb_epochs=25,verbose=True)
print(sep + "Test" + sep)
_ = get_double_accuracy(weight_share_net, test2_input, test2_target, test2_classes,verbose=True)

print("\n\n Execution done!\n")

print("NB: for a clearer overview of the meta-algorithm used to obtain these optimal parameters, please refer to the notebook explore.ipynb.\n")
