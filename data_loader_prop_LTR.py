# coding: utf-8

import random
import numpy as np
from scipy.special import softmax

class Dataloader(object):
    '''
    Usage: 
        dataloader = Dataloader(config, 'train')
        batch_data = dataloader.batch_data_iterator()
    '''

    def __init__(self, batch_size, split):
        
        self.batch_size= batch_size
        self.split = split
        self.feat_path = './data/prob/combi_' 
        self.length_path = './data/prob/length_ref'
        self.feat_dimension = 1
        self.max_len = 3136
        assert self.split in {'train', 'val', 'test'}
        
        self.combined = np.load(self.feat_path + split + '.npy',
                                allow_pickle= True).item()
        
        self.video_names = list(self.combined.keys())
        self.length = np.load(self.length_path +'.npy',
                                allow_pickle= True).item()
        
        

    def batch_data_iterator(self):
        '''
        return:
            batch_data: dict.
        '''

        while True:
            batch_feat = []  # shape: [batch_size, 32*3 + 1]
            batch_score = []  # shape: [batch_size,  1]
            random.shuffle(self.video_names)
            
            for vid in self.video_names:

                self.feat = self.combined[vid]['features']     
                self.score = self.combined[vid]['score']
                idx = np.arange(len(self.feat)) 
                random.shuffle(idx)
                
                for index in idx:
                                       
                    feat_ref = np.concatenate([self.feat[index], np.reshape(self.length[vid], [1,1])], axis= 1)
                    batch_feat.append(feat_ref)    
                    
                    if 0 <= self.score[index] <= 0.2:
                        s = 0                    
                    if 0.2 < self.score[index] <= 0.4:
                        s = 1                        
                    if 0.4 < self.score[index] <= 0.6:
                        s = 2                      
                    if 0.6 < self.score[index] <= 0.8:
                        s = 3  
                    if 0.8 < self.score[index] <= 0.9:
                        s = 4
                    if 0.9 < self.score[index] <= 1:
                        s = 5   
                      
                    batch_score.append(s)

                    if (len(batch_feat) == self.batch_size) or (index == idx[-1]):
                        
                        batch_score = softmax(batch_score)                    
                        batch_score = np.reshape(batch_score, [1, -1])
                        batch_balance_dummy = np.ones_like(batch_score) 
                      
                        if self.split == 'train':
                           yield [np.expand_dims(np.concatenate(batch_feat, axis= 0), axis= 0), np.array(batch_score), batch_balance_dummy], []
                                  
                        if self.split == 'val':
                           yield [np.expand_dims(np.concatenate(batch_feat, axis= 0), axis= 0), np.array(batch_score), batch_balance_dummy], []
                        
                        if self.split == 'test':
                           batch_balance_score_dummy = np.ones_like(batch_score) 
                           yield [np.concatenate(batch_feat, axis= 0), batch_balance_score_dummy, batch_balance_score_dummy ]
                        
                        batch_feat = []
                        batch_score = []

        
    @property
    def batch_num(self):
        return int(len(self.video_names))
