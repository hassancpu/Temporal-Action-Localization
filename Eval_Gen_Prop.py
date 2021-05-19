#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:24:02 2020

@author: user1
"""

import os
import json
import numpy as np
from scipy.interpolate import interp1d


#load the best trained model
model.load_weights('./Trained_Model/Proposal Score Prediction/Second Model'
                   '/model_best 18, v_loss=0.0009, loss=0.0006')

#import ground_truth oracle information

gt_path = './data/split_gt_info'
split_gt_dict = json.load(open(os.path.join(gt_path, 'test' + '_gt.json'), 'r'))

prob_test_ref = np.load('./data/prob/prob_test_ref.npy', allow_pickle= True).item()

# non_maximum_supression to remove redundata proposals
  
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


def getKey(item):
    return item['score']

        
if __name__=='__main__':
   
   combi_test = np.load('./data/prob/combi_test.npy'
                        , allow_pickle= True).item() 
   all_results = {}
   print ('total num:{}'.format(len(combi_test)))
   cnt = 0
   for vid in combi_test.keys():
        data = combi_test[vid]
        prob = prob_test_ref[vid]
        result = []
        cur_fps = float(split_gt_dict[vid]['fps'])
        
        tmp = [i*16 + 8 for i in range(0,len(prob[:,0]))]
        
        f_start = interp1d(tmp, prob[:,0], kind= 1, fill_value= 'extrapolate')
        f_end = interp1d(tmp, prob[:,1], kind= 1, fill_value= 'extrapolate')
        f_action = interp1d(tmp, prob[:,2], kind= 1, fill_value= 'extrapolate')
        
        start_all = f_start(range(1, tmp[-1]+ 8 + 1))
        end_all = f_end(range(1, tmp[-1]+ 8 + 1))
        action_all = f_action(range(1, tmp[-1]+ 8 + 1))
        
        dummy = np.zeros([len(data['features']), 1])
        scores = model.predict([np.concatenate(data['features'], axis= 0), dummy, dummy])
       
        
        for idx in range(len(data['features'])):
            n_start = np.argmax(start_all[int(data['proposal'][idx][0])*16:int(data['proposal'][idx][0])*16 + 16]) + 1
            n_end = np.argmax(end_all[int(data['proposal'][idx][1])*16:int(data['proposal'][idx][1])*16 + 16]) + 1
                        
            start_time = (data['proposal'][idx][0] * 16 + n_start) / cur_fps
            end_time = (data['proposal'][idx][1] * 16 + n_end) / cur_fps  
            result.append({'timestamp': [start_time, end_time], 'score': scores[idx][0]*prob[int(data['proposal'][idx][0])][0]*prob[int(data['proposal'][idx][1])][1]})  
            
        result = nms_detections(result, 0.8)
        result = sorted(result, key=getKey, reverse=True)
        cnt += 1
        
        print ('After NMS Video {} / {}: {}'.format(cnt, len(combi_test), len(result)))
        all_results[vid] = result    
        
