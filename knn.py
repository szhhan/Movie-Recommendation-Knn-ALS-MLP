#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:55:52 2019

@author: sizhenhan
"""

import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np


def load(movie,rating,timestamp):
    
    df_movies = pd.read_csv(movie,usecols=['movieId', 'title'])
    df_ratings = pd.read_csv(rating)
    
    print('Movies:')
    print(df_movies.head())
    
    df_ratings_ = df_ratings.loc[df_ratings['timestamp'] > timestamp,:]
    print('Ratings')
    print(df_ratings_.head())
    
    df_ratings = df_ratings_[['userId','movieId','rating']]
    
    num_users = len(df_ratings.userId.unique())
    num_items = len(df_ratings.movieId.unique())
    print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))
    
    return df_movies, df_ratings


def prepare(df_movies, df_ratings):
    
    movie_user_mat = df_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    movie_list =list(df_movies.set_index('movieId').loc[movie_user_mat.index].title)
    movies = {}
    for ind, movie in enumerate(movie_list):
        movies[movie] = ind 
    mat_sparse = csr_matrix(np.asarray(movie_user_mat))
    
    return mat_sparse, movies 
    
    

def match(d, fav):
    
    l = []
    for movie, ind in d.items():
        similarity = fuzz.ratio(movie.lower(), fav.lower())
        if similarity >= 60:
            l.append((movie, ind, similarity))
    l = sorted(l, key=lambda x: x[2])[::-1]
    if len(l) == 0:
        print('Not Found')
        return
    print('Your choice: {0}\n'.format([x[0] for x in l]))
    return l[0][1]

def recommend(model, data, d, fav, n):
    
    model.fit(data)
    movie_ind = match(d, fav)
    distance, ind = model.kneighbors(data[movie_ind], n_neighbors=n+1)
    distance, ind = distance.squeeze().tolist(), ind.squeeze().tolist()
    recommends = sorted(list(zip(ind,distance)), key=lambda x: x[1])[1:]
    d_rev = {}
    for k, v in d.items():
        d_rev[v] = k
    print('Recommendations for {}:'.format(fav))
    for k, v in enumerate(recommends):
        print('Rank: {0}: {1}'.format(k+1, d_rev[v[0]]))