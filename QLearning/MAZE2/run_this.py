# -*- coding:utf-8 -*- 

# @Author: Hao
# @File: run_this.py
# @Time: 2022/10/24 下午5:26
# @Describe
import torch
from DQN import DQN
from maze_env import Maze
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import early_stop

START_LEARN = 3000                          # 记忆库中达到一定数量开始学习
model_name = "model/model020.pkl"                 # 已有模型文件名称

dqn = DQN()
env = Maze()

# 如果有已经训练好的模型，加载模型，在模型基础上训练
if os.path.exists(model_name):
    dqn.eval_net = torch.load(model_name)

# 记录reward
reward_list = []
# 记录step
step_list = []
# 记录time cost
time_cost_list = []
# 记录loss
loss_list = []
# 提前终止训练标识
early_stop_flag = False
# 记录成功和失败次数
success_count = 0
fail_count = 0


for i in range(100000):
    print(f"Episode: {i}")
    # 计时开始
    start_time = time.time()
    # 重置环境
    s = env.reset()
    # 计步器
    step = 0
    # 初始化该循环对应的episode的总奖励
    episode_reward_sum = 0
    # 初始化loss（因为刚开始没没训练，没有loss）
    average_loss = []
    period_loss = 0.0

    # 开始一个episode (每一个循环代表一步)
    while True:
        step = step + 1
        # 显示实验动画
        env.render()
        # 输入该步对应的状态s，选择动作
        a = dqn.choose_action(s)
        # 执行动作，获得反馈
        s_, r, done, success_flag = env.step(a)
        # print(f"s: {s}, a: {a}, r: {r}, s_:{s_}")
        # 存储记忆
        dqn.store_transition(s, a, r, s_)
        # 积累收益
        episode_reward_sum += r

        # 更新状态
        s = s_

        # 记忆库存储达到一定数量开始学习
        if dqn.memory_counter > START_LEARN:
            loss = dqn.learn()
            average_loss.append(loss)

        if done:
            # 计算时耗
            end_time = time.time()
            time_cost = end_time - start_time

            # 记录数据
            reward_list.append(episode_reward_sum)
            time_cost_list.append(time_cost)
            step_list.append(step)
            if len(average_loss) != 0:
                period_loss = sum(average_loss) / len(average_loss)
                loss_list.append(period_loss)

            # 提前终止训练判断
            # if len(reward_list) > 101:
            #     # early_stop_flag = early_stop.early_stop_by_reward(reward_list[-200:-100], reward_list[-100:])
            #     # early_stop_flag = early_stop.early_stop_by_reward(reward_list[-100:-50], reward_list[-50:])
            #     early_stop_flag = early_stop.early_stop_by_step(step_list[-100:-50], step_list[-50:])

            print(f"episode {i}, step = {step}, cost {time_cost} s, loss = {period_loss}, reward = {episode_reward_sum}, success = {success_flag}")
            if success_flag:
                success_count += 1
            else:
                fail_count += 1
            print(f"success {success_count}, failed {fail_count}")
            break

    # if early_stop_flag:
    #     print(f"Early stop!")
    #     break

    print("--------")

# 销毁环境
env.destroy()
# 存储模型（如果早停，存储target net）
if early_stop_flag:
    torch.save(dqn.target_net, model_name)
else:
    torch.save(dqn.eval_net, model_name)

# 存储列表到文件
np.savetxt("data/reward_list3.csv", reward_list)
np.savetxt("data/time_cost_list3.csv", time_cost_list)
np.savetxt("data/step_list3.csv", step_list)
np.savetxt("data/loss_list3.csv", loss_list)

# 绘图
plt.figure(1)
plt.title("reward")
plt.xlabel("Period / i")
plt.ylabel("Reward")
plt.plot(reward_list)
plt.show()

plt.figure(2)
plt.title("time_cost")
plt.xlabel("Period / i")
plt.ylabel("Time / s")
plt.plot(time_cost_list)
plt.show()

plt.figure(3)
plt.title("step")
plt.xlabel("Period / i")
plt.ylabel("Step")
plt.plot(step_list)
plt.show()

plt.figure(4)
plt.title("loss")
plt.plot(loss_list)
plt.show()

print("OK!")
