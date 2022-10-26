# -*- coding:utf-8 -*- 

# @Author: Hao
# @File: Net.py
# @Time: 2022/10/24 上午11:20
# @Describe
import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(4, 256)
        self.hidden2 = torch.nn.Linear(256, 256)
        self.hidden3 = torch.nn.Linear(256, 128)
        self.predict = torch.nn.Linear(128, 4)

    def forward(self, observation):
        x = torch.Tensor(observation).cuda()
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.hidden3(x)
        x = torch.relu(x)
        action = self.predict(x)
        return action
