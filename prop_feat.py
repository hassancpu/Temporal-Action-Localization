#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:24:02 2020

@author: user1
"""

import os
import json
import numpy as np
import tensorflow as tf
from utils import ConfigBase
from data_loader_changed import Dataloader

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

#load the best trained model
model.load_weights('/Trained_Model/model_best 191, v_loss=0.1265, v_start=0.0226, v_end=0.0213, v_act=0.1245, v_difference=0.0013, loss=0.1214, start=0.0401, end=0.0370, act=0.1323, difference=0.0031')


#import ground_truth oracle information
split = 'train'
gt_path = './data/THUMOS14/split_gt_info'
split_gt_dict = json.load(open(os.path.join(gt_path, split + '_gt.json'), 'r'))


#define two functions to get the mean_score and vote for each unit
def get_score_matrix(score):
    score_len, recap_len = score.shape[0], score.shape[1]
    score_matrix = np.zeros((score_len + recap_len - 1, recap_len), dtype=np.float32)

    for i in range(score_len):
        for j in range(recap_len):
            if i < j:
                score_matrix[i + recap_len - 1, j] = 0.
            else:
                score_matrix[i + recap_len - 1, j] = score[i, j]
    return score_matrix[recap_len-1:]

def get_vote_and_ave(score_matrix):
    score_len, recap_len = score_matrix.shape[0], score_matrix.shape[1]

    vote_arr = np.zeros((score_len, 1), dtype=np.float32)
    ave_arr = np.zeros((score_len), dtype=np.float32)
    score_matrix_ref = np.zeros((score_len, recap_len), dtype= np.float32)
    
    for i in range(score_len):
        for j in range(recap_len):
            if i + j < score_len:
                vote_arr[i] += (score_matrix[i + j, j] >= 0.3)  # 0.7
                ave_arr[i] += score_matrix[i + j, j]
                score_matrix_ref[i, j] = score_matrix[i+j, j]
        ave_arr[i] /= min(score_len - i, recap_len)

    return vote_arr, ave_arr, score_matrix_ref

def nms_detections(proposals, overlap=0.7):
    """Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    proposals: list of item, each item is a dict containing 'timestamp' and 'score' field

    Returns
    -------
    new proposals with only the proposals selected after non-maximum suppression.
    """

    if len(proposals) == 0:
        return proposals

    props = np.array([item['timestamp'] for item in proposals])
    scores = np.array([item['score'] for item in proposals])
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    # 改进
    area = (t2 - t1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]

    out_proposals = []
    for idx in range(nms_props.shape[0]):
        prop = nms_props[idx].tolist()
        score = float(nms_scores[idx])
        out_proposals.append({'timestamp': prop, 'score': score})
    return out_proposals

def calc_iou(gt_starts, gt_ends, anchor_start, anchor_end):
    '''
    calc intersection over union, frame_level
    gt_starts, gt_ends: multi values, shape: [gt_num]
    anchor_start, anchor_end: single value
    '''
    and_starts = np.maximum(gt_starts, anchor_start)
    and_ends = np.minimum(gt_ends, anchor_end)
    and_lens = np.maximum(and_ends - and_starts, 0)
    
    or_starts = np.minimum(gt_starts, anchor_start)
    or_ends = np.maximum(gt_ends, anchor_end)
    or_lens = np.maximum(or_ends - or_starts, 0)

    iou =  and_lens / np.asarray(or_lens, dtype= np.float32)
    return iou


def getKey(item):
    return item['score']

#set the parameters

class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()

        self.split_gt_info_path = './data/THUMOS14/split_gt_info'
        self.batch_size = 1
        self.window_size = 1000000
        self.recap_length = 17
        self.feat_path_rgb = './data/THUMOS14/thumos14_i3d_features_rgb_with_ucf101.hdf5'
        self.feat_path_flow = './data/THUMOS14/thumos14_i3d_features_flow_with_ucf101.hdf5'
        self.feat_resolution = 16

        self.feat_mode = 'both'
        if self.feat_mode == 'rgb' or self.feat_mode == 'flow':
            self.feat_dimension = 1024
        else:
            self.feat_dimension = 2048

        self.balance_ratio = [1, 1, 1]
        
        
if __name__=='__main__':
   config = Config()

   dataloader = Dataloader(config, split)
   data_iterator = dataloader.batch_data_iterator()
   
   all_results = {}
   all_prob = {}
   all_vote = {}
   print ('total num:{}'.format(dataloader.batch_num))
   
   for vid_idx in range(dataloader.batch_num):
        data = data_iterator.__next__()
        
        data[1] = np.zeros_like(data[1])
        scores = model.predict([data[0], data[1], data[2]])

        scores = np.reshape(scores,[-1,3,config.recap_length])    
        
        _start_scores = scores[:,0,:]
        _end_scores = scores[:,1,:]
        _action_scores =  scores[:,2,:]
        
        cur_video_name = data[3]

        start_score_matrix = get_score_matrix(_start_scores)
        end_score_matrix = get_score_matrix(_end_scores)
        action_score_matrix = get_score_matrix(_action_scores)

        start_vote, start_ave, start_score_matrix_ref = get_vote_and_ave(start_score_matrix)
        end_vote, end_ave, end_score_matrix_ref = get_vote_and_ave(end_score_matrix)
        action_vote, action_ave, action_score_matrix_ref = get_vote_and_ave(action_score_matrix)      
        
        score_matrix_ref = np.concatenate((start_score_matrix_ref, end_score_matrix_ref,
                                           action_score_matrix_ref), axis= -1)
        vote_matrix_ref = np.concatenate((start_vote/17, end_vote/17, action_vote/17), axis= -1)
        
        vote_V = 1
        start_idx_part = np.where(start_vote>=vote_V)[0]
        end_idx_part = np.where(end_vote>=vote_V)[0]

        start_idx = list(start_idx_part)
        end_idx = list(end_idx_part)

        for i in range(1, len(start_ave) - 1):
            if start_ave[i] > start_ave[i - 1] and start_ave[i] > start_ave[i + 1]:
                if i not in start_idx:
                    start_idx.append(i)

        for i in range(1, len(end_ave) - 1):
            if end_ave[i] > end_ave[i - 1] and end_ave[i] > end_ave[i + 1]:
                if i not in end_idx:
                    end_idx.append(i)

        feat = split_gt_dict[cur_video_name]['gt_feat_stamps']
        feat = np.array(feat)
        
        candi_results = []
        for i in range(len(start_idx)):
            candi_start = start_idx[i]
            cnt = 0
            for j in range(len(end_idx)):
                candi_end = end_idx[j]
                if candi_end > candi_start:
                    cnt += 1
                    if cnt <= 20:
                        info_density = np.max(calc_iou(feat[:,0], feat[:,1]
                                             ,candi_start, candi_end))                                             
                        candi_results.append([candi_start, candi_end, info_density])
          
        candi_results = np.asarray(candi_results, dtype=np.float32)
        
        all_results[cur_video_name] = candi_results        
        all_prob[cur_video_name] = score_matrix_ref
        all_vote[cur_video_name] = vote_matrix_ref
        
np.save('./data/THUMOS14/prob/' + split + '.npy', all_prob, allow_pickle= True)

#np.save('/home/user1/Master Thesis/Python/My Self/data/THUMOS14/prob/val_vote.npy'
#        , all_vote, allow_pickle= True)

np.save('./data/THUMOS14/proposals/' + split + '.npy', all_results, allow_pickle= True)        
        
