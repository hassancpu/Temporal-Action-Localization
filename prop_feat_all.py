#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 10:18:59 2020

@author: user1
"""
from scipy.interpolate import interp1d
import numpy as np

split = 'train'

prop_train = np.load('./data/proposals/' + split + '.npy', allow_pickle= True).item()
prob_train = np.load('./data/prob/' + split + '.npy', allow_pickle= True).item()

prob_train_ref = {}
for vid in prob_train.keys():
    prob = prob_train[vid]
    prob_ref = np.zeros([prob.shape[0], 3])
    for i in range(prob.shape[0]):
        prob_ref[i,0] = np.mean(prob[i,:17])
        prob_ref[i,1] = np.mean(prob[i,17:34])
        prob_ref[i,2] = np.mean(prob[i,34:])
    prob_train_ref[vid] = prob_ref    


num_sample_start=8
num_sample_end=8
num_sample_action=16
num_sample_interpld = 3
    
cnt = 0
combi_train = {}
for vid in prop_train.keys():
    pros = prop_train[vid]
    feat = prob_train_ref[vid]
    
    video_scale = len(feat)
    combi_train_temp = []
    video_extend = int(video_scale / 4 + 10)
    
    tmp_zeros=np.zeros([video_extend]) 
    
    score_start = feat[:,0]
    score_start=np.concatenate((tmp_zeros,score_start,tmp_zeros))

    score_end = feat[:,1]
    score_end=np.concatenate((tmp_zeros,score_end,tmp_zeros))
    
    score_action = feat[:,2]
    score_action=np.concatenate((tmp_zeros,score_action,tmp_zeros))
    
    
    tmp_cell = 1
    tmp_x = [-tmp_cell -(video_extend-1-ii)*tmp_cell for ii in range(video_extend) ] + \
             [tmp_cell +ii*tmp_cell - 1 for ii in range(video_scale)] + \
              [tmp_cell + feat.shape[0] +ii*tmp_cell - 1 for ii in range(video_extend)]
    
    f_start = interp1d(tmp_x,score_start,axis=0)
    f_end = interp1d(tmp_x,score_end,axis=0)
    f_action = interp1d(tmp_x,score_action,axis=0)
    
    for i in range(len(pros)):
        xmin = pros[i][0]
        xmax = pros[i][1]
        xlen = pros[i][1] - pros[i][0]
        xmin_0=xmin-xlen/5
        xmin_1=xmin+xlen/5
        xmax_0=xmax-xlen/5
        xmax_1=xmax+xlen/5   
        
        #start
        plen_start= (xmin_1-xmin_0)/(num_sample_start-1)
        plen_sample = plen_start / num_sample_interpld
        tmp_x_new = [ xmin_0 - plen_start/2 + plen_sample * ii for ii in range(num_sample_start*num_sample_interpld +1 )] 
        
        tmp_y_new_start_start=f_start(tmp_x_new)
        tmp_y_new_start_s = [np.mean(tmp_y_new_start_start[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_start) ]
      
        tmp_y_new_start_end=f_end(tmp_x_new)
        tmp_y_new_start_e = [np.mean(tmp_y_new_start_end[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_start) ]
      
        tmp_y_new_start_action=f_action(tmp_x_new)
        tmp_y_new_start_a = [np.mean(tmp_y_new_start_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_start) ]
        
        #end
        plen_end= (xmax_1-xmax_0)/(num_sample_end-1)
        plen_sample = plen_end / num_sample_interpld
        tmp_x_new = [ xmax_0 - plen_end/2 + plen_sample * ii for ii in range(num_sample_end*num_sample_interpld +1 )] 

        tmp_y_new_end_start=f_start(tmp_x_new)
        tmp_y_new_end_s = [np.mean(tmp_y_new_end_start[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_end) ]
                
        tmp_y_new_end_end=f_end(tmp_x_new)
        tmp_y_new_end_e = [np.mean(tmp_y_new_end_end[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_end) ]
        
        
        tmp_y_new_end_action=f_action(tmp_x_new)
        tmp_y_new_end_a = [np.mean(tmp_y_new_end_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_end) ]
        
        #action
        plen_action= (xmax-xmin)/(num_sample_action-1)
        plen_sample = plen_action / num_sample_interpld
        tmp_x_new = [ xmin - plen_action/2 + plen_sample * ii for ii in range(num_sample_action*num_sample_interpld +1 )] 

        tmp_y_new_action_start=f_start(tmp_x_new)
        tmp_y_new_action_s = [np.mean(tmp_y_new_action_start[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_action) ]

        tmp_y_new_action_end=f_end(tmp_x_new)
        tmp_y_new_action_e = [np.mean(tmp_y_new_action_end[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_action) ]
        
        tmp_y_new_action_action=f_action(tmp_x_new)
        tmp_y_new_action_a = [np.mean(tmp_y_new_action_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_action) ]

        
        tmp_y_new_start = np.concatenate((tmp_y_new_action_s, tmp_y_new_start_s, tmp_y_new_end_s))
        tmp_y_new_end = np.concatenate((tmp_y_new_action_e, tmp_y_new_start_e, tmp_y_new_end_e))
        tmp_y_new_action = np.concatenate((tmp_y_new_action_a, tmp_y_new_start_a, tmp_y_new_end_a))

        tmp_feature = np.concatenate([tmp_y_new_action,tmp_y_new_start,tmp_y_new_end])
        tmp_feature = np.reshape(tmp_feature, [-1, 32*3])
        
        combi_train_temp.append([tmp_feature, pros[i][0], pros[i][1], pros[i][2]])
   
    combi_train_temp = np.array(combi_train_temp)
    
    p_temp = []
    for i in range(len(list(combi_train_temp[:, 1]))):
        p_temp.append([list(combi_train_temp[:, 1])[i], list(combi_train_temp[:, 2])[i]])
        
    combi_train[vid] = {'features': list(combi_train_temp[:, 0]), 'proposal': p_temp
    , 'score': list(combi_train_temp[:, 3])} 
    cnt +=1        
    print(cnt)
    

np.save('./data/prob/combi_' + split + '.npy',combi_train, allow_pickle= True)    
