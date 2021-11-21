import numpy as np
import os
import tensorflow as tf
import pickle

from client import Client

from models import *
# from data import *
from model_params import *
import time
from plot import *

DEBUG = False
# import matplotlib as plt

# tf.debugging.set_log_device_placement(True)


#########################
import pandas as pd
import numpy as np

BASE_DIR = "/Users/krishna/Documents/IoT-IDS-using-Federated-Learning-main-2/Datasets/AE_formed_data/"
LABEL_DIR = "/Users/krishna/Documents/IoT-IDS-using-Federated-Learning-main-2/Datasets/AE_formed_data/"


DATA_LINK = [
    'CICIDS 2018',
    'CICIDS 2017',
    'BOT IOT',
    'NSL_KDD',
    'TON_IOT',
    'UNSW_NB15'
]

NUM_LABELS = [
    12,
    15,
    4,
    5,
    10,
    10
]


def split_data(DEBUG=False):  # to split it into multiple rounds
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train_all = []
    y_train_all = []

    x_test_all = []
    y_test_all = []

    #     pca = []

    for i in range(len(DATA_LINK)):
        train = pd.read_csv(BASE_DIR + DATA_LINK[i] + "_train.csv")
        test = pd.read_csv(BASE_DIR + DATA_LINK[i] + "_valid.csv")
        if DEBUG:
            train = train.sample(320)
            test = test.sample(64)

        ytrain = np.load(LABEL_DIR +  f"{DATA_LINK[i]}_train.npy")
        ytest = np.load(LABEL_DIR + f"{DATA_LINK[i]}_test.npy")
        xtrain = train.iloc[:, :-NUM_LABELS[i]]
        xtest = test.iloc[:, :-NUM_LABELS[i]]
        assert ytrain.shape[0] == xtrain.shape[0]
        assert ytest.shape[0] == xtest.shape[0]

        x_train_all.append(np.asarray(xtrain))
        x_test_all.append(np.asarray(xtest))

        y_train_all.append(np.asarray(ytrain))
        y_test_all.append(np.asarray(ytest))

    return x_train_all, x_test_all, y_train_all, y_test_all



def model_average(client_weights):
    average_weight_list = []
    for index1 in range(len(client_weights[0])):  # -2 to exclude softmax dense
        layer_weights = []
        for index2 in range(len(client_weights)):
            weights = client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight = np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list


def create_model():
    model = get_model()
    ann_weight = model.get_weights()
    return ann_weight


CLIENT_PRINT = {
    0: "CICIDS 2018",
    1: "CICIDS 2017",
    2:  "BOT IOT",
    3: "NSL_KDD",
    4:"TON_IOT",
    5:"UNSW_NB15"
}

PARAMS = get_model_params()
dump_number=[100,200,300,400,500]

def train_server(training_rounds, epoch, batch, learning_rate):
    accuracy_list = [[],[],[],[],[],[]]
    client_weight_for_sending = []

    one_epoch_time=0 
    
    x_data,x_test,y_data,y_test = split_data(DEBUG = DEBUG)
    for index1 in range(1, training_rounds):
        
        #calcualting time
        if index1==1:
            start_time=time.time()
        
        if index1==2:
            end_time=time.time()
            one_epoch_time=end_time-start_time
        
        print('Time Remaining is ', ((training_rounds-index1)*one_epoch_time)/3600, 'Hours')
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged = []
        for index in range(len(y_data)):
            print('-------Client-------', CLIENT_PRINT[index])
            if index1 == 1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight= create_model()
                client = Client(
                        x_data[index],
                        y_data[index],
                        epoch,
                        learning_rate,
                        initial_weight,
                        batch,
                    PARAMS
                    )
                MLP_weights = client.train()
                client_weights_tobe_averaged.append(MLP_weights)
            else:
                client = Client(
                            x_data[index],
                                y_data[index],
                                epoch,
                                learning_rate,
                                client_weight_for_sending[index1 - 2],
                                batch,
                                PARAMS
                               )
                MLP_weights = client.train()
                client_weights_tobe_averaged.append(MLP_weights)

        client_average_weight= model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)
        with open(f'FL_round_{index1}.txt', 'wb') as f:
                pickle.dump(client_average_weight, f)
        if index1 != 1:
            os.remove(f'FL_round_{index1 - 1}.txt')
        print(f"Evaluation for round{index1}:")
        # model = get_model(PARAMS,
        #                   mlp_weights=client_average_weight,
        #                   )
        
        model=get_model()
        model.set_weights(client_average_weight)

        model.compile(
            loss=[
                tf.keras.losses.BinaryCrossentropy()
            ],
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            ]
        )
        for index in range(len(y_test)):
            result = model.evaluate(x_test[index], [y_test[index]])
            accuracy = result
            print(f"###### Accuracy for {CLIENT_PRINT[index]} -> {result}")
            accuracy_list[index].append(accuracy[1])
        plot1=Plot(accuracy_list,index1)
        plot1.plot_figure()
        del plot1
        
        if index1 in dump_number:
            with open(f"accuracy_list_{index1}.txt", "wb") as output_file:
                pickle.dump(accuracy_list, output_file)
                
            with open(f"client_weight_for_sending_{index1}.txt", "wb") as output_file:
                pickle.dump(client_weight_for_sending, output_file)
                
            
            
        
    return accuracy_list


if __name__ == '__main__':
    with tf.device('/GPU:0'):
        training_accuracy_list = train_server(
                                                    training_rounds=500,
                                                    epoch=3,
                                                    batch=32,
                                                    learning_rate=0.001
                                                 )
        with open('accuracy_list.txt','wb') as fp:
            pickle.dump(training_accuracy_list,fp)



