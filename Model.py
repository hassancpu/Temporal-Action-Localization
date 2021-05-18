#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hassan Keshvari Khojasteh
"""

# import the necessary libraries

import tensorflow as tf
import keras.backend as k
from utils import ConfigBase
from keras.models import Model
import keras.regularizers as reg
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from data_loader import Dataloader
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Input, GRU, Dense, Multiply, Masking, Lambda, Concatenate

#define the network

hidden_units = 513
recap_len = 17
feat_dim = 2048
units_start = 171
units_end = 171
units_action = 171

mask = Masking(mask_value= 0, input_shape= (None,feat_dim))
gru_1 = GRU(units= hidden_units, return_sequences= True, dropout= 0.2, 
            kernel_regularizer= reg.l2(1e-5))          
gru_2 = GRU(units= hidden_units, return_sequences= True, dropout= 0.2, 
            kernel_regularizer= reg.l2(1e-5))

d_1 = Dense(recap_len, activation = 'sigmoid', kernel_regularizer=reg.l2(1e-5))
d_2 = Dense(recap_len, activation = 'sigmoid', kernel_regularizer=reg.l2(1e-5))
d_3 = Dense(recap_len, activation = 'sigmoid', kernel_regularizer=reg.l2(1e-5))
       
#deifne the network structure

batch_feat = Input(shape= (None, 2048))  
x = mask(batch_feat)
x = gru_1(x)

x = gru_2(x)

x_1 = Lambda(lambda x: x[:, :, : units_start])(x)
x_2 = Lambda(lambda x: x[:, :, units_start: units_start+ units_end])(x)
x_3 = Lambda(lambda x: x[:, :, units_start+ units_end: ])(x)

score_start = d_1(x_1)     
score_end = d_2(x_2)       
score_action = d_3(x_3)   
scores = Concatenate(axis= -1)([score_start, score_end, score_action])     

#loss  
    
scores_true = Input(shape = (None, recap_len*3))
batch_balanced_mask = Input(shape = (None, recap_len*3)) 

scores_true_ref = Multiply()([scores_true,  batch_balanced_mask])
scores_true_ref = tf.reshape(scores_true_ref, [-1,3,recap_len])
start_scores_true = scores_true_ref[:, 0, :]
end_scores_true = scores_true_ref[:, 1, :]
action_scores_true = scores_true_ref[:, 2, :]


scores_predicted = scores
scores_predicted = Lambda(lambda x:x, output_shape= lambda s:s)(scores_predicted)
scores_predicted = Multiply()([scores_predicted,  batch_balanced_mask])
scores_predicted = tf.reshape(scores_predicted, [-1,3,recap_len])    
start_scores_pred = scores_predicted[:,0,:]
end_scores_pred = scores_predicted[:,1,:]
action_scores_pred = scores_predicted[:,2,:]   
  
     
loss_start = k.binary_crossentropy(start_scores_true, 
                                   start_scores_pred)
loss_end = k.binary_crossentropy(end_scores_true,
                                 end_scores_pred)
loss_action = k.binary_crossentropy(action_scores_true,
                                    action_scores_pred)
 
loss_start = k.mean(loss_start)
loss_end = k.mean(loss_end)
loss_action = k.mean(loss_action)
 

# Now compute total loss

loss_ = loss_start + loss_end + 0.2*loss_action 
      
#loss_ = k.mean(loss_)
   
model = Model(inputs= [batch_feat, scores_true,batch_balanced_mask],
              outputs= scores)

model.add_loss(loss_)

model.add_metric(loss_start, 'start')
model.add_metric(loss_end, 'end')
model.add_metric(loss_action, 'action')

# Visualize the model

model.summary()
#plot_model(model, show_shapes= True)


#compile 

model.compile(optimizer= Adam(3e-3))
    


def step_decay(epoch, lr):
    if (epoch+1) % 51 == 0 and epoch:
        return lr*0.94
    return lr

learning_rate_new = LearningRateScheduler(step_decay)

#set the parameters
     
class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()

        self.split_gt_info_path = './data/split_gt_info'
        self.batch_size = 32
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

#Save the trained model after each epoch

class CustomSaver(Callback):
  def on_epoch_end(self, epoch, accuracy, logs= {}):
    # save the trained model after each epoch
    model.save("./Trained_Model/model {}, v_loss={:6.4f}, v_start={:6.4f}"
               ", v_end={:6.4f}, v_act={:6.4f}, loss={:6.4f}, start={:6.4f}"
               ", end={:6.4f}, act={:6.4f}".format(epoch+1, accuracy['val_loss']
                , accuracy['val_start'], accuracy['val_end'], accuracy['val_action']
                , accuracy['loss'], accuracy['start'], accuracy['end'], accuracy['action']))

Saver = CustomSaver()   

                                                               

                    
#Now, Start Training!
if __name__=='__main__': 
    
    config = Config()
    train_loader= Dataloader(config, 'train')
    val_loader= Dataloader(config, 'val')   
    train_gen = train_loader.batch_data_iterator()
    val_gen = val_loader.batch_data_iterator()
    train_step = train_loader.batch_num
    val_step = val_loader.batch_num


    
    history = model.fit_generator(train_gen, epochs= 210, validation_data= val_gen, steps_per_epoch= train_step,
                        validation_steps= val_step,callbacks= [Saver, learning_rate_new],
                        verbose= 1)
    
    #Plot the desired metrics during training
    
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('Loss')
    plt.xlabel('num_of_epochs')
    plt.ylabel('binary_crossentropy_loss')
    plt.legend(['val', 'train'], loc = 'upper right')
    plt.show()
    
    plt.plot(history.history['val_start'])
    plt.plot(history.history['start'])
    plt.title('Loss Start')
    plt.xlabel('num_of_epochs')
    plt.ylabel('binary_crossentropy_loss_start')
    plt.legend(['val', 'train'], loc = 'upper right')
    plt.show()
        
    
    
 
