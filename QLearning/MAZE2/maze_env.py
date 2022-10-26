# -*- coding:utf-8 -*- 

# @Author: Hao
# @File: maze_env.py
# @Time: 2022/10/22 上午10:36
# @Describe: 迷宫环境
import numpy as np
import time
import tkinter as tk

UNIT = 40  # 像素
MAZE_H = 10  # 网格高度
MAZE_W = 10  # 网格宽度


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    # 构建迷宫
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 创建网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 起始点
        origin = np.array([20, 20])
        # 终止点
        over = np.array([100, 100])

        # 失败格子1
        hell1_center = over + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # 失败格子2
        hell2_center = over + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        # 失败格子3
        hell3_center = over + np.array([UNIT * 3, UNIT * 3])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        # 失败格子4
        hell4_center = over + np.array([UNIT * 2, UNIT * 4])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')
        # 失败格子5
        hell5_center = over + np.array([UNIT * 4, UNIT * 2])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')

        # 创建椭圆形，代表成功
        oval_center = over + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 创建红色矩形，代表agent
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 打包画布
        self.canvas.pack()

    # 更新画布，返回当前位置
    def reset(self):
        self.update()
        # time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # 返回当前环境坐标
        return self.canvas.coords(self.rect)

    # 红色方块移动
    def step(self, action):
        # 获取当前位置信息
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # Agent移动

        s_ = self.canvas.coords(self.rect)  # 获取下一状态

        # 成功 or 失败 Flag
        success_flag = False

        # 奖励策略
        if s_ == self.canvas.coords(self.oval):
            reward = 10
            done = True
            success_flag = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3), self.canvas.coords(self.hell4), self.canvas.coords(self.hell5)]:
            reward = -10
            done = True
        else:
            reward = -0.1
            done = False

        # 返回 下一状态，奖励和是否结束
        return s_, reward, done, success_flag

    # 更新画布
    def render(self):
        # time.sleep(0.1)
        self.update()
