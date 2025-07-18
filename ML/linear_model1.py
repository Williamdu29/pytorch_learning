import torch

# print(torch.__version__)
import numpy as np
import matplotlib.pyplot as plt

# 线性模型

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
#人工数据集

def forward(x):
    return x*w
#前向传播函数，也就是我们要找的模型

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

w_list=[]
mse_list=[]
for w in np.arange(0.0,4.1,0.1): #穷举法的实现
    print("w=",w)
    loss_sum=0
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val=forward(x_val)
        loss_val=loss(x_val,y_val)
        loss_sum+=loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    #计算损失值和预测值
    
    print("MSE=",loss_sum/3)
    w_list.append(w)
    mse_list.append(loss_sum/3)
    
plt.plot(w_list,mse_list)
plt.ylabel("LOSS")    
plt.xlabel("w")
plt.show()  