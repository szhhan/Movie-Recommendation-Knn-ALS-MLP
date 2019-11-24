#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:28:16 2019

@author: sizhenhan
"""

from pyspark.mllib.recommendation import ALS
import math

def train_ALS(train, valid, iters, regs, ranks):
    
    error_min = float('inf')
    rank_best = -1
    reg_best = 0
    model_best = None
    for rank in ranks:
        for reg in regs:
            model = ALS.train(ratings=train, iterations=iters,rank=rank,lambda_=reg,seed=42)
            valid_d = valid.map(lambda p: (p[0], p[1]))
            pred = model.predictAll(valid_d).map(lambda x: ((x[0], x[1]), x[2]))
            combine = valid.map(lambda x: ((x[0], x[1]), x[2])).join(pred)

            MSE = combine.map(lambda x: (x[1][0] - x[1][1])**2).mean()
            err = math.sqrt(MSE)
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, err))
            if err < error_min:
                error_min = err
                rank_best = rank
                reg_best = reg
                model_best = model
    print('\nThe best model has {} latent factors and regularization = {}'.format(rank_best, reg_best))
    return model_best


def find_id(movies, favs):
    l = []
    for movie in favs:
        tmp = movies.filter(movies.title.like('%{}%'.format(movie))).select('movieId').rdd .map(lambda r: r[0]).collect()
        l.extend(tmp)
    return list(set(l))


def add_user(train, l, spark_context):
    id_new = train.map(lambda x: x[0]).max() + 1
    max_rating = train.map(lambda x: x[2]).max()
    row = [(id_new, movie, max_rating) for movie in l]
    rdd = spark_context.parallelize(row)
    return train.union(rdd)

def get_test(train, movies, l):
    id_new = train.map(lambda x: x[0]).max() + 1
    return movies.rdd.map(lambda x: x[0]).distinct().filter(
        lambda x: x not in l).map(lambda x: (id_new, x))

def recommendation(model_best_params, ratings, movies, favs, n, spark_context):
    
    l = find_id(movies, favs)
    train_new = add_user(ratings, l, spark_context)
    
    model = ALS.train(ratings=train_new, iterations=model_best_params.get('iterations', None),
        rank=model_best_params.get('rank', None),lambda_=model_best_params.get('lambda_', None),seed=42)
    
    test = get_test(ratings, movies, l)
    
    preds = model.predictAll(test).map(lambda x: (x[0],x[1], x[2]))
    
    top_n = preds.sortBy(lambda x: x[2], ascending=False).take(n)
    top_id = [x[1] for x in top_n]
    
    return movies.filter(movies.movieId.isin(top_id)).select('title').rdd.map(lambda x: x[0]).collect()
