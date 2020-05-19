# -*- coding: utf-8 -*-
"""
Created on Fri May 1 00:00:00 2020

@author: Yann
"""
# hyperoptimization functions for MNIST training
import torch
from time import perf_counter 
from misc_funcs import compute_time
from opt import Kfold_CVdouble
from misc_funcs import RANDOM_SEED,EXPLORE_K,BIG_K,NB_EPOCHS,sep
import copy
import random
import pickle


class Param():
    """
    class representing a hyper-parameters combination for grid search purpose
    Parameters:
    -----------
    arch : nn.Model
        a non-trained instance of the architecture to use
    loss:
        loss to use
    """
    hyper_params = None
    rando = lambda lista: lista[torch.randint(len(lista),(1,)).item()]
    @staticmethod
    def set_hyper_params(hyper_params):
        Param.hyper_params = hyper_params
    @staticmethod
    def check_hyper():
        """check that hyperparameters are indeed defined
        """
        if Param.hyper_params is None:
            raise NameError("Hyper params are not defined in Param class")
    @staticmethod
    def parse(classi):
        """parse name for str representation"""
        return str(classi).split('.')[-1].split("'")[0]

    def __init__(self,arch=None,loss_comp=None,loss_class=None,optimizer=None,lr=None,lambd_=None):
        if None in [arch,loss_comp,loss_class,optimizer,lr,lambd_]:
            self.params = self.generate_rand_params()
        else:
            self.params = {"arch":arch,
                           "loss_comp":loss_comp,
                           "loss_class":loss_class,
                           "optim":optimizer,
                           "lr":lr,
                           "lambda":lambd_}
        self.scores = []
        self.score_mean = -1
        self.individuality = -1
        
    def get_params(self):
        return list(self.params.values())
    
    
    def generate_rand_params(self):

        """
        allocates random values to the current parameters
        """
        self.check_hyper()
        rand_values = [Param.rando(param) for param in Param.hyper_params]
        names = ["arch","comp_loss","class_loss","optim","lr","lambda"]
        return {name:val for name,val in zip(names,rand_values)}
    
    def mutate(self):
        """
        mutate a parameter at random
        """
        self.check_hyper()
        #choses a parameter at random
        param_indx = torch.randint(len(Param.hyper_params),(1,)).item()
        values = Param.hyper_params[param_indx]
        
        #choses a value at random
        value_indx = torch.randint(len(values),(1,)).item()
        value = values[value_indx]
        
        key = list(self.params.keys())[param_indx]
        #reassign to the random value
        self.params[key] = value
    
    def KFold(self,train2_input,train2_target,train2_classes,nb_epochs=NB_EPOCHS,K=5,verbose=False,prog_bar=False):
        """
        computes KFold on the train set passed in argument for the current parameter value
        """
        scores = Kfold_CVdouble(train2_input,train2_target,train2_classes,
                                *self.get_params(),nb_epochs=nb_epochs,K=K,verbose=verbose,prog_bar=prog_bar)
        self.set_scores(scores)
        return scores
        
    def set_scores(self,scores):
        self.scores = scores.tolist()
        self.score_mean = scores.mean().item()
        self.score_std = scores.std().item()
                
    def __str__(self):
        returned = "{}_{}_{}_{}_{}_{}_#ind#_{:.2f}_#score#_{:.2f}".format(Param.parse(self.params["arch"]),
                                               Param.parse(self.params["loss_comp"]),
                                               Param.parse(self.params["loss_class"]),
                                               Param.parse(self.params["optim"]),
                                               self.params["lr"],
                                               self.params["lambda"],
                                               self.individuality,
                                               self.score_mean)
        return returned
    def __repr__(self):
        return str(self)

def GetNTop(params,N):
    """
    return N best parameters
    Parameters:
    ----------
    params : list(Param)
        params to extract a top from
    """
    return sorted(params,key = lambda x:x.score_mean)[::-1][:N]

def GetMax(params):
    """return best parameter"""
    return max(params,key = lambda x:x.score_mean)

class HyperGrid():
    """
    Hypergrid represents the hyperspace in which we explore parameters combinations
    """
    def __init__(self,Archis,CompLoss,ClassLoss,Optimizers,LRs,Lambdas,save_path):
        self.data = torch.empty(len(Archis),len(CompLoss),len(ClassLoss),len(Optimizers),len(LRs),len(Lambdas)).tolist()
        for a,archi in enumerate(Archis):
            for b,comp_loss in enumerate(CompLoss):
                for c,class_loss in enumerate(ClassLoss):
                    for d,optim_ in enumerate(Optimizers):
                        for e,lr in enumerate(LRs):
                            for f,lambd_ in enumerate(Lambdas):
                                self.data[a][b][c][d][e][f] = Param(archi,comp_loss,class_loss,optim_,lr,lambd_)
        self.save_path= save_path
    def lin_view(self):
        """linear representation of a hyper grid"""
        linHGRID = [f for a in self.data for b in a for c in b for d in c for e in d for f in e]
        return linHGRID
    
    def estimate_time(self,train2_input,train2_target,train2_classes,K):
        """
        estimate the time it would take to run a full grid search with the current hypergrid
        """
        random.seed(5)
        trash_param = copy.deepcopy(random.choice(self.lin_view()))
        t1 = perf_counter()
        trash_param.KFold(train2_input,train2_target,train2_classes,K)
        t2 = perf_counter()
        delta = t2 - t1
        tot = delta * len(self.lin_view())
        print()
        print(compute_time(tot))
        
    def save(self):
        """
        save the hypergrid with all its parameters
        """
        torch.save(self,self.save_path, pickle_module=pickle)
    