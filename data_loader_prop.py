# coding: utf-8

import random
import numpy as np

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
        self.feat_dimension = 1
        assert self.split in {'train', 'val', 'test'}
        
        self.combined = np.load(self.feat_path + split + '.npy',
                                allow_pickle= True).item()
        
        self.video_names = list(self.combined.keys())
        self.feat = []
        self.score = []
        for i in self.video_names:
            self.feat.append(self.combined[i]['features'])
            self.score.append(self.combined[i]['score'])
        
        self.feat = np.concatenate(self.feat, axis= 0)
        self.score = np.concatenate(self.score, axis= 0)
        
        self.split_size = len(self.feat)
        
        
    def balance_mask_func(self, score, ratio= 2):
        '''
        score: interpolated action scores, shape: [batch_size,1]
        '''


        def balance_mask(score, cur_ratio):
            score_flat = np.reshape(score, [-1])
            thres_score = (score_flat >= 0.7).astype(np.float32)
            pos = np.where(thres_score == 1.)[0]
            neg = np.where(thres_score == 0.)[0]
            sample_idx = list(pos) + random.sample(list(neg), min(int(len(pos) * cur_ratio), len(neg)))
            mask = np.zeros_like(score_flat, dtype=np.float32)
            mask[sample_idx] = 1.
            return mask

        balanced_mask = balance_mask(score, ratio)


        return balanced_mask
    
 

    def batch_data_iterator(self):
        '''
        return:
            batch_data: dict.
        '''

        if self.split == 'train':
            r = list(zip(self.feat, self.score))
            random.shuffle(r)
            self.feat, self.score = zip(*r)

        while True:
            batch_feat = []  # shape: [batch_size, 32*3]
            batch_score = []  # shape: [batch_size,  1]
            self.feat = np.array(self.feat)
            self.score = np.array(self.score)
            idx = np.arange(len(self.feat))
            
            for index in idx:

    
                batch_feat.append(self.feat[index])      
                batch_score.append(self.score[index])                
                                       
                if (len(batch_feat) == self.batch_size) or (index == idx[-1]):
                    
                    batch_balance_score_mask = self.balance_mask_func(batch_score, 1)
                    if self.split == 'train':
                       yield [np.concatenate(batch_feat, axis= 0), np.array(batch_score), batch_balance_score_mask ], []
                              
                    if self.split == 'val':
                       yield [np.concatenate(batch_feat, axis= 0), np.array(batch_score), batch_balance_score_mask ], []
                    
                    if self.split == 'test':
                       batch_score_dummy = np.ones_like(batch_score) 
                       yield [np.concatenate(batch_feat, axis= 0), batch_score_dummy, batch_balance_score_mask ]
                    
                    batch_feat = []
                    batch_score = []
            
            if self.split == 'train':
               r = list(zip(self.feat, self.score))
               random.shuffle(r)
               self.feat, self.score = zip(*r)

    @property
    def batch_num(self):
        return int(np.ceil(float(self.split_size) / self.batch_size))


