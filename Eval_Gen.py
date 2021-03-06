#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:24:02 2020

@author: Hassan Keshvari Khojasteh
"""
import numpy as np
from utils import ConfigBase
from data_loader import Dataloader



#load the best trained model
model.load_weights('./Trained_Model/model_Best 202, v_loss=0.1220, v_start=0.0223, v_end=0.0210,'
                   'v_act=0.1108, loss=0.1233, start=0.0414, end=0.0370, act=0.1405')

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

    vote_arr = np.zeros((score_len), dtype=np.float32)
    ave_arr = np.zeros((score_len), dtype=np.float32)

    for i in range(score_len):
        for j in range(recap_len):
            if i + j < score_len:
                vote_arr[i] += (score_matrix[i + j, j] >= 0.3)  # 0.7
                ave_arr[i] += score_matrix[i + j, j]
        ave_arr[i] /= min(score_len - i, recap_len)

    return vote_arr, ave_arr

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
    # ??????
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

def getKey(item):
    return item['score']

class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()

        self.split_gt_info_path = './data/split_gt_info'
        self.batch_size = 1
        self.window_size = 100
        self.recap_length = 17
        self.feat_path_rgb = './data/thumos14_i3d_features_rgb_with_ucf101.hdf5'
        self.feat_path_flow = './data/thumos14_i3d_features_flow_with_ucf101.hdf5'
        self.feat_resolution = 16

        self.feat_mode = 'both'
        if self.feat_mode == 'rgb' or self.feat_mode == 'flow':
            self.feat_dimension = 1024
        else:
            self.feat_dimension = 2048

        self.balance_ratio = [1, 1, 1]
        self.out_json_dir = './Proposals/prop.json'
        

if __name__=='__main__':
   config = Config()

   dataloader = Dataloader(config, 'test')
   data_iterator = dataloader.batch_data_iterator()
   
   all_results = {}
   all_prob = {}
   print ('total num:{}'.format(dataloader.batch_num))
   
   for vid_idx in range(dataloader.batch_num):
        data = data_iterator.__next__()
        
        dummy = np.zeros_like(data[1])
        scores = model.predict([data[0], dummy, data[2]])
        
        scores = np.reshape(scores,[-1,3,config.recap_length])    
        
        _start_scores = scores[:,0,:]
        _end_scores = scores[:,1,:]
        _action_scores =  scores[:,2,:]
        
        cur_video_name = data[3]
        cur_fps = float(dataloader.split_gt_dict[cur_video_name]['fps'])

        start_score_matrix = get_score_matrix(_start_scores)
        end_score_matrix = get_score_matrix(_end_scores)
        action_score_matrix = get_score_matrix(_action_scores)

        start_vote, start_ave = get_vote_and_ave(start_score_matrix)
        end_vote, end_ave = get_vote_and_ave(end_score_matrix)
        action_vote, action_ave = get_vote_and_ave(action_score_matrix)      
        
        
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
        
        result = []
        candi_results = []
        for i in range(len(start_idx)):
            candi_start = start_idx[i]
            cnt = 0
            for j in range(len(end_idx)):
                candi_end = end_idx[j]
                if candi_end > candi_start:
                    cnt += 1
                    if cnt <= 20:
                        info_density = np.sum(action_ave[candi_start: candi_end + 1]) / float((candi_end - candi_start + 1)) * start_ave[candi_start] * end_ave[candi_end]                                              
                        start_time = (candi_start * 16 + 8) /cur_fps
                        end_time = (candi_end * 16 + 8) /cur_fps
                        candi_results.append([start_time, end_time, info_density])
                        result.append(
                            {'timestamp': [start_time, end_time], 'score': info_density})
        if not result:
            result.append({'timestamp': [0., dataloader.split_gt_dict[cur_video_name]['duration']], 'score': 0.2})

        candi_results = np.asarray(candi_results, dtype=np.float32)
        print ('{}/{}: {}, gt_num: {}'.format(vid_idx + 1, len(dataloader.video_names), cur_video_name, len(dataloader.split_gt_dict[cur_video_name]['gt_frame_stamps'])))
        print ('Before NMS: {}'.format(len(candi_results)))

        result = nms_detections(result, overlap=0.8)
        result = sorted(result, key=getKey, reverse=True)

        print ('After NMS: {}'.format(len(result)))
        all_results[cur_video_name] = result
        
