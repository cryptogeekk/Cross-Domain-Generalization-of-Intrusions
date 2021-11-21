#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 23:22:54 2021

@author: krishna
"""

class Plot:

    def __init__(self, accuracy_list, index):
        self.accuracy_list=accuracy_list
        self.index=index
   
    def plot_figure(self):
        import matplotlib.pyplot as plt
        import numpy as np
        
        plot_list=[10,30,50,70,100,120,150,200,250,300,350,400,450]
        if self.index in plot_list:
            length=len(self.accuracy_list)
            epochs = np.arange(0,self.index)
            

            plt.plot(epochs, self.accuracy_list[0],  label='CICIDS 2018')
            plt.plot(epochs, self.accuracy_list[1],  label='CICIDS 2017')
            plt.plot(epochs, self.accuracy_list[2],  label='BOT IOT')
            plt.plot(epochs, self.accuracy_list[3],  label='NSL_KDD')
            plt.plot(epochs, self.accuracy_list[4],  label='TON_IOT')
            plt.plot(epochs, self.accuracy_list[5],  label='UNSW_NB15')
    
            plt.title('Federated Learning Accuracy')
            plt.xlabel('Communication Round')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        # else:
        #     plt.plot(epochs, self.accuracy_list[0+(6)*(self.index-1)],  label='CICIDS 2018')
        #     plt.plot(epochs, self.accuracy_list[1+(6)*(self.index-1)],  label='CICIDS 2017')
        #     plt.plot(epochs, self.accuracy_list[2+(6)*(self.index-1)],  label='BOT IOT')
        #     plt.plot(epochs, self.accuracy_list[3+(6)*(self.index-1)],  label='NSL_KDD')
        #     plt.plot(epochs, self.accuracy_list[4+(6)*(self.index-1)],  label='TON_IOT')
        #     plt.plot(epochs, self.accuracy_list[5+(6)*(self.index-1)],  label='UNSW_NB15')
    
        #     plt.title('Federated Learning Accuracy')
        #     plt.xlabel('Communication Round')
        #     plt.ylabel('Accuracy')
        #     plt.grid(True)
        #     plt.legend()
        #     plt.show()
       


# epochs = np.arange(1,22)
 
# accuracy_list=training_accuracy_list
# plt.plot(epochs, accuracy_list[0],  label='CICIDS 2018')
# plt.plot(epochs, accuracy_list[1],  label='CICIDS 2017')
# plt.plot(epochs, accuracy_list[2],  label='BOT IOT')
# plt.plot(epochs, accuracy_list[3],  label='NSL_KDD')
# plt.plot(epochs, accuracy_list[4],  label='TON_IOT')
# plt.plot(epochs, accuracy_list[5],  label='UNSW_NB15')

# plt.title('Federated Learning Accuracy')
# plt.xlabel('Communication Round')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.legend()
# plt.show()


# for index in range(0,20):
#     for index1 in range(len(accuracy_list)):
#         accuracy_list[index1].append(np.random.randint(90))


# np.random.randint(90)

