#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:55:45 2021

@author: krishna
"""

def get_model_params():
    params = {
    'num_columns' : 80,
    'num_labels': 56,
    'hidden_units' :[128, 128, 1024, 512, 512, 256],
    'dropout_rates' :[0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43] ,
    }

    return params


  model = get_model(params_file = self.params,
                           mlp_weights = self.weights
                           )
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                loss=[
                    tf.keras.losses.BinaryCrossentropy()
                ],
                metrics=[
                    'accuracy',
                ]
                )

  history = model.fit(
      self.dataset_x, self.dataset_y,
      epochs=self.epoch_number,
      batch_size=self.batch

  )

#cicids 2017  
data=pd.read_csv('/Users/krishna/Documents/IoT-IDS-using-Federated-Learning-main-2/Datasets/AE_formed_data/CICIDS 2017_train.csv')

columns=data.columns
y_columns=columns[80:95]
train_y=data[[x for x in y_columns ]]
data.drop([x for x in y_columns], axis=1, inplace=True)


test_data=pd.read_csv('/Users/krishna/Documents/IoT-IDS-using-Federated-Learning-main-2/Datasets/AE_formed_data/CICIDS 2017_valid.csv')
y_valid=test_data[[x for x in y_columns ]]
test_data.drop([x for x in y_columns], axis=1, inplace=True)
X_valid=test_data



from tensorflow import keras
#sparse_categorical_crossentropy
#binary_crossentropy

model=keras.models.Sequential([
            keras.layers.Flatten(input_shape=[80,]),
            keras.layers.Dense(200,activation='tanh'),
            keras.layers.Dense(100,activation='tanh'),
            keras.layers.Dense(15,activation='sigmoid')
            ])

 
# model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

history=model.fit(data,train_y,epochs=5,batch_size=32) 




model1.compile()
validation_data=(X_valid,y_valid)

###Ton_IoT
data=pd.read_csv('/Users/krishna/Documents/IoT-IDS-using-Federated-Learning-main-2/Datasets/AE_formed_data/TON_IOT_train.csv')

columns=data.columns
y_columns=columns[80:90]
train_y=data[[x for x in y_columns ]]
data.drop([x for x in y_columns], axis=1, inplace=True)


test_data=pd.read_csv('/Users/krishna/Documents/IoT-IDS-using-Federated-Learning-main-2/Datasets/AE_formed_data/TON_IOT_valid.csv')
y_valid=test_data[[x for x in y_columns ]]
test_data.drop([x for x in y_columns], axis=1, inplace=True)
X_valid=test_data


from tensorflow import keras
#sparse_categorical_crossentropy
#binary_crossentropy

model=keras.models.Sequential([
            keras.layers.Flatten(input_shape=[80,]),
            keras.layers.Dense(200,activation='tanh'),
            keras.layers.Dense(100,activation='tanh'),
            keras.layers.Dense(10,activation='sigmoid')
            ])

 
# model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

history=model.fit(data,train_y,epochs=20,batch_size=32, validation_data=(X_valid,y_valid)) 

"""Looping starts from here"""

base_dir='/Users/krishna/Documents/IoT-IDS-using-Federated-Learning-main-2/Datasets/AE_formed_data/'
train_dataset_list=['BOT IOT_train.csv', 'CICIDS 2018_train.csv', 'NSL_KDD_train.csv', 'UNSW_NB15_train.csv', 'CICIDS 2017_train.csv', 'TON_IOT_train.csv']
test_dataset_list=['BOT IOT_valid.csv', 'CICIDS 2018_valid.csv', 'NSL_KDD_valid.csv', 'UNSW_NB15_valid.csv', 'CICIDS 2017_valid.csv', 'TON_IOT_valid.csv']

import os 
from tensorflow import keras



for index in range(len(train_dataset_list)):
    print('Training for ', train_dataset_list[index])
    path_train=base_dir+train_dataset_list[index]
    path_test=base_dir+test_dataset_list[index]
    
    data=pd.read_csv(path_train)

    columns=data.columns
    y_columns=columns[80:len(columns)]
    train_y=data[[x for x in y_columns ]]
    data.drop([x for x in y_columns], axis=1, inplace=True)


    test_data=pd.read_csv(path_test)
    y_valid=test_data[[x for x in y_columns ]]
    test_data.drop([x for x in y_columns], axis=1, inplace=True)
    X_valid=test_data
    


    
    
    

