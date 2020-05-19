# -*- coding: utf-8 -*-
"""
Created on Fri May 1 00:00:00 2020

@author: Yann
"""
# optimization functions for MNIST training

# testing purpose
#RANDOM_SEED = 14
# development purpose
RANDOM_SEED = 3
EXPLORE_K = 3
BIG_K = 10
NB_EPOCHS = 25
sep = "-" * 20


intervals = (
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )

def compute_time(seconds, gran=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:gran])
