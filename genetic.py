# -*- coding: utf-8 -*-
"""
Created on Fri May 1 00:00:00 2020

@author: Yann
"""
# Genetic Algorithm for parameter optimization
from hyperopt import Param
from misc_funcs import RANDOM_SEED,EXPLORE_K,BIG_K,NB_EPOCHS,sep
import matplotlib.pyplot as plt
import copy
import torch


def generate_population(n_pop=100):
    """ generate the initial population"""
    population = [Param() for _ in range(n_pop)]
    return population

def compute_individuality(population):
    """
    computes the individuality score of each individual in the population 
    """
    count_matrix = [{i:0 for i in param} for param in Param.hyper_params]
    for ind in population:
        for val,dico in zip(ind.params.values(),count_matrix):
            if not val in dico:
                dico[val] = 0
            dico[val] += 1
    for ind in population:
        diffs = []
        for val,dico in zip(ind.params.values(),count_matrix):
            diffs.append(sum(dico.values()) - dico[val])
        indiv = (1/(sum([1/diff if diff!=0 else 6 for diff in diffs])))
        ind.individuality = indiv
    individualities = [ind.individuality for ind in population]
    maxo,mino = max(individualities),min(individualities)
    if maxo != mino:
        for ind in population:
            ind.individuality = (ind.individuality - mino)/(maxo-mino)

def compute_fitness(train2_input,train2_target,train2_classes,population,K=5,verbose=False):
    """compute the fitness of each individual in the population"""
    for ind in population:
            if ind.score_mean == -1:
                ind.KFold(train2_input,train2_target,train2_classes,K=K,verbose=verbose)

def selection(population,selec_ratio=0.5,lambd_ =0.2):
    """carries out a selection in the population by keeping only selec_ratio of the population
    sort the remaining parameters
    """
    #sort population
    top_decreas = sorted(population,key=lambda x: x.score_mean + lambd_ * x.individuality)[::-1]
    top_num = int((len(population)*selec_ratio))
    if top_num < 2:
        raise NameError("Not enough individual for breeding! Increase selec_ratio or n_pop.")
    return top_decreas[:top_num]    

def breed(top_pop,n_pop=100,chance=0.1):
    """
    create some new individuals from the population
    Parameters:
    -----------
    top_pop : list(Param)
        reduced population (best ratio)
    n_pop : int
        size to reach for the returned population
    n_chance : 
    """
    n_chance = int(1/chance)
    assert(n_pop >= len(top_pop))
    top_pop = copy.deepcopy(top_pop)
    n_top = len(top_pop)
    n_miss = n_pop - n_top
    for i in range(n_miss):
        parindx1,parindx2 = torch.randperm(n_top).tolist()[:2]
        par1 = top_pop[parindx1]
        par2 = top_pop[parindx2]
        params1 = par1.get_params()
        params2 = par2.get_params()
        n_params = len(params1)
        bits = torch.randint(2,(n_params,)).tolist()
        params = [par1 if bit == 0 else par2 for par1,par2,bit in zip(params1,params2,bits)]
        child = Param(*params)
        n_rand = torch.randint(n_chance,(1,)).item()
        if n_rand == 0:
            print("mutation!")
            child.mutate()
        top_pop.append(child)
    return top_pop

def plot_population(population):
    """
    plot the population
    """
    N = len(population)
    fig,ax = plt.subplots(1)
    ax.set_ylim([0,1])
    ax.bar(range(N),[ind.score_mean for ind in population])

if __name__ == "__main__":
    ####################################
    # size of the population
    N_POP = 5
    # number of iteration of selection
    N_ITER = 5
    # selection ratio
    SELEC_RATIO = 0.6
    # chance of mutation
    CHANCE = 0.1
    # number of fold for evaluation
    K = 2
    ####################################
    # we keep a param to receive
    best_indiv = Param()
    #initial population
    population = generate_population(N_POP)
    compute_fitness(population,K=2,verbose=True)
    compute_individuality(population)
    for i in range(N_ITER):
        clear_output(wait=False)
        print("Population Progression: {} %".format(i/N_ITER * 100))
        plot_population(population)
        plt.pause(0.05)
        population = selection(population,selec_ratio=SELEC_RATIO)
        if population[0].score_mean > best_indiv.score_mean:
            best_indiv = population[0]
        population = breed(population,n_pop=N_POP,chance=CHANCE)
        compute_fitness(population,K=3)
        compute_individuality(population)