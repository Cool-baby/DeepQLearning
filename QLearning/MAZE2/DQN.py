# -*- coding:utf-8 -*- 

# @Author: Hao
# @File: DQN.py
# @Time: 2022/10/24 上午11:27
# @Describe
import torch
from Net import Net
import numpy as np

MEMORY_CAPACITY = 10000                          # 记忆库容量
BATCH_SIZE = 32                                 # 批处理样本大小
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
EPSILON = 0.9                                   # 贪心策略
GAMMA = 0.9                                     # 奖励衰减

device = "cuda:0"


class DQN:

    # 初始化神经网络
    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        self.learn_step_counter = 0  # 初始化计步器
        self.memory_counter = 0  # 初始化记忆库计步器
        self.memory = np.zeros((MEMORY_CAPACITY, 4 * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0001)
        self.loss_func = torch.nn.SmoothL1Loss()

    # 选择行动（训练过程中贪心探索）
    def choose_action(self, x):
        x = torch.unsqueeze(torch.tensor(x), 0)
        # 贪心
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            # 获取收益最大的action
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            # print(f"<0.9: {action}, max = {torch.max(torch.Tensor(actions_value))}, actions_value = {actions_value}")
        # 探索
        else:
            action = np.random.randint(0, 4)
            # print(f">0.9: {action}")
        return action

    # 预测行动
    def predict_action(self, x):
        x = torch.unsqueeze(torch.tensor(x), 0)
        actions_value = self.eval_net.forward(x)
        print(actions_value)
        # 获取收益最大的action
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        # 一组记忆
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 学习
    def learn(self):
        # 目标网络参数更新(每隔100次更新一次目标网络)
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的数据批训练
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # 将BATCH_SIZE个s抽出
        b_s = torch.Tensor(b_memory[:, :4])
        # 将BATCH_SIZE个a抽出
        b_a = torch.LongTensor(b_memory[:, 4:5].astype(int)).to(device)
        # 将BATCH_SIZE个r抽出
        b_r = torch.Tensor(b_memory[:, 5:6]).to(device)
        # 将BATCH_SIZE个s_抽出萨的
        b_s_ = torch.Tensor(b_memory[:, -4:])

        # 获取BATCH_SIZE个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        # eval_net(b_s)通过评估网络输出BATCH_SIZE行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_eval = self.eval_net.forward(b_s).to(device).gather(1, b_a)
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出BATCH_SIZE行每个b_s_对应的一系列动作值
        q_next = self.target_net(b_s_).detach()
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为BATCH_SIZE的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # 输入BATCH_SIZE个评估值和BATCH_SIZE个目标值，使用均方损失函数
        loss = self.loss_func(q_eval, q_target)
        # 记录loss
        loss_number = loss.data.cpu().numpy()

        # 清空上一步的残余更新参数值
        self.optimizer.zero_grad()
        # 误差反向传播, 计算参数更新值
        loss.backward()
        # 更新评估网络的所有参数
        self.optimizer.step()

        return loss_number
