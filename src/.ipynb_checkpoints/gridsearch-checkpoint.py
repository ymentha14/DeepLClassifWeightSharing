# -*- coding: utf-8 -*-
"""
Created on Fri May 1 00:00:00 2020

@author: Yann
"""
# GridSearch parameters
from IPython.display import clear_output
from hyperopt import HyperGrid
from misc_funcs import RANDOM_SEED,EXPLORE_K,BIG_K,NB_EPOCHS,sep


def GridSearch(HG,train2_input,train2_target,train2_classes,K=EXPLORE_K):
    """carries out a grid search on the combination passed in argument
    Parameters:
    -----------
    HG : HyperGrid
    """
    #TODO: remove the [:1]
    linGRID = HG.lin_view()
    for i,param in enumerate(linGRID):
        clear_output(wait=True)
        print("Grid Search progression: {:.2f} %".format(i/len(linGRID)*100))
        param.KFold(train2_input,train2_target,train2_classes,K=K)
        #backup in case of crash
        HG.save()

    print("Grid Search done! Hyperparam saved.")