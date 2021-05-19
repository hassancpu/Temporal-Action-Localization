#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: user1
"""

# import the necessary libraries

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from data_loader_prop_LTR import Dataloader
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Input, Dense, Dropout, Multiply, Activation, Lambda

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

#define the network

feat_dim = 97
units_out = 1

d_1 = Dense(feat_dim, activation = 'relu')
drop_1 = Dropout(0.1)

d_2 = Dense(int(feat_dim/2), activation = 'relu')
drop_2 = Dropout(0.1)
d_3 = Dense(units_out, activation = 'relu')
       
#deifne the network structure

batch_feat = Input(shape= (feat_dim, ))  

x = d_1(batch_feat)
x = drop_1(x)
x = d_2(x)
x = drop_2(x)
score = d_3(x)


#loss  
       
score_true = Input(shape = (1, ))
score_balance_mask = Input(shape= (1, ))

score_true_ref = Multiply()([score_true,  score_balance_mask])
score_true_ref= Lambda(lambda x : tf.reshape(x, (1, -1)))(score_true_ref)

score_ref = Multiply()([score, score_balance_mask])
score_reshape = Lambda(lambda x : tf.reshape(x, (1, -1)))(score_ref)
score_softmax = Activation('softmax')(score_reshape)

# Now, compute total loss and metrics

loss = -tf.reduce_mean(tf.reduce_sum(score_true_ref * tf.log(score_softmax)))

   
model = Model(inputs= [batch_feat, score_true, score_balance_mask],
              outputs= score)

model.add_loss(loss)

# Visualize the model

model.summary()

#compile 

model.compile(optimizer= Adam(1e-3))
    


def step_decay(epoch, lr):
    if (epoch+1) % 10 == 0 and epoch:
        return lr*0.1
    return lr

learning_rate_new = LearningRateScheduler(step_decay)
     
#Save the trained model after each epoch

class CustomSaver(Callback):
  def on_epoch_end(self, epoch, accuracy, logs= {}):
    # save the trained model after each epoch
    if epoch +1 > 1:
        model.save("/Trained_Model/Proposal Score Prediction/Second Model"
                   "/model {}, v_loss={:6.4f}, loss={:6.4f}".format(epoch+1
                   , accuracy['val_loss'], accuracy['loss']))

Saver = CustomSaver()   
                                                               
                
#Now, Start Training!

if __name__=='__main__': 
    
    train_loader= Dataloader(3000000, 'train')
    val_loader= Dataloader(3000000, 'val')   
    train_gen = train_loader.batch_data_iterator()
    val_gen = val_loader.batch_data_iterator()
    train_step = train_loader.batch_num
    val_step = val_loader.batch_num

    
    history = model.fit_generator(train_gen, epochs= 20, validation_data= val_gen, steps_per_epoch= train_step,
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
    
