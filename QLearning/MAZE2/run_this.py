# -*- coding:utf-8 -*- 

# @Author: Hao
# @File: run_this.py
# @Time: 2022/10/24 下午5:26
# @Describe
import torch
from DQN import DQN
from maze_env import Maze
import os

START_LEARN = 1000                          # 记忆库中达到一定数量开始学习

dqn = DQN()
env = Maze()

# 如果有已经训练好的模型，加载模型，在模型基础上训练
if os.path.exists("model002.pkl"):
    dqn.eval_net = torch.load("model001.pkl")

for i in range(200):
    print(f"Episode: {i}")
    # 重置环境
    s = env.reset()
    # 计步器
    step = 0
    # 初始化该循环对应的episode的总奖励
    episode_reward_sum = 0

    # 开始一个episode (每一个循环代表一步)
    while True:
        step = step + 1
        # 显示实验动画
        env.render()
        # 输入该步对应的状态s，选择动作
        a = dqn.choose_action(s)
        # 执行动作，获得反馈
        s_, r, done = env.step(a)
        # print(f"s: {s}, a: {a}, r: {r}, s_:{s_}")
        # 存储记忆
        dqn.store_transition(s, a, r, s_)
        # 积累收益
        episode_reward_sum += r

        # 更新状态
        s = s_

        # 记忆库存储达到一定数量开始学习
        if dqn.memory_counter > START_LEARN:
            dqn.learn()

        if done:
            print(f"episode {i}, step = {step}, reward sum = {episode_reward_sum}")
            break

    print("--------")

print("OK!")
env.destroy()
torch.save(dqn.eval_net, "model001.pkl")
