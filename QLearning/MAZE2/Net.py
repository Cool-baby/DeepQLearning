# -*- coding:utf-8 -*- 

# @Author: Hao
# @File: Net.py
# @Time: 2022/10/24 上午11:20
# @Describe
import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(4, 50)
        self.hidden.weight.data.normal_(0, 0.1)
        self.predict = torch.nn.Linear(50, 4)
        self.predict.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        action = self.predict(x)
        return action
