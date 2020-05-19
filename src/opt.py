# -*- coding: utf-8 -*-
"""
Created on Fri May 1 00:00:00 2020

@author: Yann
"""
# Optimization of simple network for MNIST digit recognition

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F

from IPython.display import clear_output


from misc_funcs import RANDOM_SEED,EXPLORE_K,BIG_K,NB_EPOCHS,sep

class Net2(nn.Module):
    """
    Reference architecture to use for the project digit classification
    Parameters:
    -----------
    n_hidden : int
        number of hidden units
    chan : int 
        number of input images to deal with
    """
    def __init__(self,n_hidden = 100,chan = 1):
        super(Net2,self).__init__()
        self.hidden = n_hidden
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(chan,32,kernel_size=3),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.MaxPool2d(kernel_size=2,stride=2)
            ,nn.BatchNorm2d(64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256,n_hidden),
           # nn.Dropout(0.5),
            nn.Linear(n_hidden,10)
            #nn.Softmax2d()
        )
    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

def get_accuracy(model,inputs,targets,set_type=""):
    """
    compute the accuracy for the trained model on the data passed in parameters
    Parameters:
    -----------
    model : nn.Module
    inputs : torch.Tensor (N x 1 x 14 x 14)
        input data to compute the accuracy for
    targets: torch.Tensor (N)
        corresponding targets
    """
    assert(inputs.size(0) == targets.size(0))
    tot_loss = 0
    nb_correct = 0
    batch_size = 20
    for train,target in zip(inputs.split(batch_size),
                           targets.split(batch_size)):
        pred = model(train)
        pred = torch.argmax(pred,axis = 1)
        nb_correct += (pred == target).int().sum().item()
    accuracy = nb_correct /inputs.size(0)
    print("{} accuracy: {:.2f}".format(set_type,accuracy) )
    return accuracy

def Kfold_CV(inputs,targets,architecture,criterion,optimizer,lr,K=BIG_K,
             nb_epochs=NB_EPOCHS,verbose=False):
    """
    train the model passed in parameter on the data
    Parameters:
    -----------
    architecture : nn.Module architecture
        the architecture to do the Kfold on
    inputs : torch.Tensor (N x 1 x 14 x 14)
        input data to compute the accuracy for
    targets: torch.Tensor (N)
        corresponding targets
    K : int
        number of folds
    """ 
    if RANDOM_SEED is not None:
        torch.manual_seed(RANDOM_SEED)
    assert(K>=2)
    N = inputs.size(0)
    indxes = torch.randperm(N)\
                  .split(int(N/K))
    accs = torch.empty(K)
    for k in range(K):
        model = architecture()
        
        test_indx = indxes[k]
        train_indx = torch.cat((indxes[:k]+indxes[k+1:]),0)
        
        train_inp,train_targ = inputs[train_indx],targets[train_indx]
        test_inp,test_targ = inputs[test_indx],targets[test_indx]
        train_model(train_inp,train_targ,model,criterion,optimizer,lr,nb_epochs,verbose)
        acc = get_accuracy(model,test_inp,test_targ)
        accs[k] = acc
    clear_output()
    print(sep)
    print("Accuracies for {}-fold:{}".format(K,accs.tolist()))
    print("Accuracy:{:.2f} +- {:.2f}".format(accs.mean(),accs.std()))
    print(sep)
    return accs.tolist()

def train_model(train_input,train_target,model,criterion,optimizer,
                lr,nb_epochs=NB_EPOCHS,verbose=False):
    """
    train the model passed in parameter on the data
    Parameters:
    -----------
    model : nn.Module
    train_input : torch.Tensor (N x 1 x 14 x 14)
        input data to compute the accuracy for
    train_target: torch.Tensor (N)
        corresponding targets
    nb_epochs : int
    """
    if RANDOM_SEED is not None:
        torch.manual_seed(RANDOM_SEED)
    optimizer = optimizer(model.parameters(), lr)
    batch_size = 100
    for e in range(nb_epochs):
        if verbose:
            clear_output(wait=True)
            print("Progression:{:.2f} %".format(e/nb_epochs*100))
        for inputs,targets in zip(train_input.split(batch_size),
                            train_target.split(batch_size)):
            output = model(inputs)
            loss = criterion(output,targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()

class Clonable(nn.Module):
    """
    SuperClass allowing cloning of an instance while still carrying out weight randomization
    """

    def __init__(self, *args):
        self.params = args
        super(Clonable, self).__init__()

    def clone(self):
        return type(self)(*self.params)


class Naive(Clonable):
    """
    Naive model comparing trivially the output of 2 reference models in order to assess which digit is greater 
    Trains the 2 models to recognize their respectiv digits, no comparison involved
    """

    def __init__(self):
        super(Naive, self).__init__()
        self.net0 = Net2()
        self.net1 = Net2()

    def forward(self, x):
        x0 = self.net0(x[:, 0].unsqueeze(1))
        x1 = self.net1(x[:, 1].unsqueeze(1))
        comp = (x0.max(1)[1] <= x1.max(1)[1]).int()
        ret = torch.FloatTensor(comp.size(0), 2).zero_()
        ret.scatter_(1, comp.long().unsqueeze(1), 1)
        return x0, x1, ret

    def __str__(self):
        return "NaiveArch"


class WeightAux(Clonable):
    """
    Architecure implementing weight sharing and/or auxiliary loss
    Parameters:
    -----------
    weightshare : Bool
        number 
    auxloss : Bool 
        number of input images to deal with
    """

    def __init__(self, weightshare=True, auxloss=True):
        super(WeightAux, self).__init__(weightshare, auxloss)
        # applies weightsharing
        self.weightshare = weightshare
        # triggers the use of an auxiliary loss in train_double_model when set to True
        self.auxloss = auxloss
        self.net0 = Net2()
        self.net1 = Net2()
        self.linblock = nn.Sequential(
            nn.Linear(20, 40), nn.ReLU(), nn.Linear(40, 80), nn.ReLU(), nn.Linear(80, 2)
        )

    def forward(self, x):
        x0 = self.net0(x[:, 0].unsqueeze(1))
        x1 = (
            self.net0(x[:, 1].unsqueeze(1))
            if self.weightshare
            else self.net1(x[:, 1].unsqueeze(1))
        )
        comp = torch.cat((x0, x1), dim=1)
        comp = self.linblock(comp)
        return x0, x1, comp

    def __str__(self):
        stro = "Arch_"
        if self.weightshare:
            stro += "Weightshare"
        if self.auxloss:
            stro += "Auxloss"
        if not self.weightshare and not self.auxloss:
            stro += "classic"
        return stro


def get_double_accuracy(model, train_input, train_target, train_classes, verbose=False):
    """
    computes the accuracy for a double model, that is, a model which computes digit comparison
    Parameters:
    -----------
    model : nn.Module
        a torch neural net able to compute digit comparison on the mnist dataset
    train_input : torch.Tensor (N x 2 x 14 x 14)
        Mnist digits train set
    train_target: torch.Tensor (N)
        tensor containing either 0 or 1 values: 0 if the first digit is greater than the second, 1 otherwise
    train_classes: (N x 2)
        classes of the digits in train_input
    """
    assert train_input.size(0) == train_target.size(0)
    N = train_input.size(0)
    tot_loss = 0
    nb_correct = 0
    batch_size = 20

    # given a prediction powre and the target, output the number of correctly classified samples
    add_res = (
        lambda pred, target: (torch.argmax(pred, axis=1) == target).int().sum().item()
    )

    score0 = 0
    score1 = 0
    scorecomp = 0

    for inputs, comp_targs, classes in zip(
        train_input.split(batch_size),
        train_target.split(batch_size),
        train_classes.split(batch_size),
    ):
        targ0 = classes[:, 0]
        targ1 = classes[:, 1]
        x0, x1, comp = model(inputs)

        score0 += add_res(x0, targ0)
        score1 += add_res(x1, targ1)
        scorecomp += add_res(comp, comp_targs)

    acc0 = score0 / N
    acc1 = score1 / N
    acc_comp = scorecomp / N

    if verbose:
        print("Accuracy 1st Network: {:^10.2f}".format(acc0))
        print("Accuracy 2nd Network: {:^10.2f}".format(acc1))
        print("Accuracy comparison: {:^12.2f}".format(acc_comp))
    return acc0, acc1, acc_comp


def train_double_model(train_input, train_target, train_classes,model,crit_comp,crit_class,optimizer,
                       lr,lambd_=0.75,nb_epochs=NB_EPOCHS,verbose=False,prog_bar=False):
    """
    train a model on the given train_input using the passed parameters
    Parameters:
    -----------
    model: nn.Module   (3 arch)
        the model to train 
    crit_comp:         (2 types)               
        the criterion for comparison 
    optimizer:         (3 types)
        the chosen optimizer
    lr: float          (4 types)
        learning rate                 
    lambd_: float      (3 values)
        the parameter balancing the 2 losses:loss =  lambd_ * comp_loss + (1-lambd_) * class_loss 
    """
    if RANDOM_SEED is not None:
        torch.manual_seed(RANDOM_SEED)
    if crit_comp is not None:
        crit_comp = crit_comp()
    optimizer = optimizer(model.parameters(), lr=lr)

    batch_size = 100
    crit_class = crit_class()

    for e in range(nb_epochs):
        if prog_bar:
            clear_output(wait=True)
            print("Progression:{:.2f} %".format(e / nb_epochs * 100))
        for inputs, comp_targs, classes in zip(
            train_input.split(batch_size),
            train_target.split(batch_size),
            train_classes.split(batch_size),
        ):
            
            targ0 = classes[:, 0]
            targ1 = classes[:, 1]
            x0, x1, comp = model(inputs)
            
            def get_good_loss(crit,data,target,n_els):
                if isinstance(crit, (nn.CrossEntropyLoss, nn.NLLLoss)):
                    loss = crit(data, target)
                else:
                    y_onehot = torch.FloatTensor(target.size(0), n_els).zero_()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, target.unsqueeze(1), 1)
                    loss = crit(data, y_onehot)
                return loss
            
            if (isinstance(model, Naive) or model.auxloss):
                loss_class = get_good_loss(crit_class,x0,targ0,10) + get_good_loss(crit_class,x1,targ1,10)
            if (not isinstance(model,Naive)):
                loss_comp = get_good_loss(crit_comp,comp,comp_targs,2)
            
            if isinstance(model, Naive):
                totloss = loss_class            
            elif model.auxloss:
                totloss = lambd_ * loss_comp + (1 - lambd_) * loss_class
            else:
                totloss = loss_comp
            model.zero_grad()
            totloss.backward()
            optimizer.step()


def Kfold_CVdouble(inputs,targets,classes,model_template,crit_comp,crit_class,optimizer,
                   lr,lambd_=0.75,K=4,nb_epochs=NB_EPOCHS,verbose=False,prog_bar=False):
    """
    runs K fold Cross validation on the data passed in parameter
    Args:
        model_template: the type of architecture for classif (3 arch)
        crit_comp: the criterion for comparison (2 sorts)
        optimizer: the chosen optimizer (3 types)
        lr: learning rate (4 types)
        lambd_: ratio lambd_ * comp_loss + (1-lambd_) * class_loss
    """
    if RANDOM_SEED is not None:
        torch.manual_seed(RANDOM_SEED)
    assert K >= 2
    N = inputs.size(0)
    indxes = torch.randperm(N).split(int(N / K))
    accs = torch.empty(K, 3)
    for k in range(K):
        if prog_bar:
            clear_output(wait=True)
            print("Progression {:.2f}%".format(k/K*100))
        model = model_template.clone()

        test_indx = indxes[k]
        train_indx = torch.cat((indxes[:k] + indxes[k + 1 :]), 0)

        train_inp = inputs[train_indx]
        train_targ = targets[train_indx]
        train_classes = classes[train_indx]

        test_inp = inputs[test_indx]
        test_targ = targets[test_indx]
        test_classes = classes[test_indx]

        train_double_model(train_inp,train_targ,train_classes,model,crit_comp,crit_class,optimizer,
                           lr,lambd_,nb_epochs=nb_epochs)
        res = get_double_accuracy(model, test_inp, test_targ, test_classes)
        # 0th column: 1st group acc 1th column 2nd group acc 3rd column comp accuracy
        accs[k] = torch.Tensor(res)
    if verbose:
        clear_output()
        mean_std = lambda x: (x.mean().item(), x.std().item())
        print(sep + "Validation Accuracies for {}-fold:".format(K) + sep)
        print("Accuracy 1st network: {:.2f} +- {:.2f}".format(*mean_std(accs[:, 0])))
        print("Accuracy 2nd network: {:.2f} +- {:.2f}".format(*mean_std(accs[:, 1])))
        print("Accuracy comparison:  {:.2f} +- {:.2f}".format(*mean_std(accs[:, 2])))
    return accs[:, 2]