# project    :don't to try
# fileName   :QLearning-2dims.py
# user       :cheng
# createDate :2021-1-27 20:14
'''
地图如下：
■●□□□
□□□□●
□□●□□
●□□●□
□□●★□
■为起点，★为终点，●为坑
'''
import copy
import time
import numpy as np
import pandas as pd
import os

np.random.seed(int(time.time()))
N_STATES = 5
holes = [[0, 1], [1, 4], [2, 2], [3, 0], [3, 3], [4, 2]]
reward = [4, 3]
EPSILON = 0.9
ALPHA = 0.1  # 学习率
LAMBDA = 0.9  # 衰减系数
ACTIONS = ['up', 'left', 'down', 'right']
MAX_EPISODES = 10000
DELAY = False
SHOW_STEP = False


def delay(n, option=True):
    if option:
        time.sleep(n)
    else:
        pass


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states * n_states, len(actions))),  # q_table 全 0 初始
        columns=actions,  # columns 对应的是行为名称
        index=[(col, row) for col in range(N_STATES) for row in range(N_STATES)]
    )
    return table


def print_env(env_list, m, n, display=True):
    if display:
        for i in range(m):
            for j in range(n):
                print(env_list[i][j], end='')
            print()
    else:
        pass


def update_env(S, episode, step_counter):
    env_list = [['□' for c in range(N_STATES)] for r in range(N_STATES)]
    for hole in holes:
        env_list[hole[0]][hole[1]] = '●'
    env_list[reward[0]][reward[1]] = '★'
    for hole in holes:
        if S == hole:
            interaction = '第%s回合: 掉入坑中%s，总步数=%s' \
                          % (episode + 1, list(map(lambda x: x + 1, S)), step_counter)
            print('\r{}'.format(interaction), end='')
            env_list[S[0]][S[1]] = '■'
            print_env(env_list, N_STATES, N_STATES, False)
            delay(2, DELAY)
            os.system('cls')
            return
    if S == reward:
        interaction = '第%s回合: 成功到达终点%s，总步数=%s' \
                      % (episode + 1, list(map(lambda x: x + 1, S)), step_counter)
        print('\r{}'.format(interaction), end='')
        print_env(env_list, N_STATES, N_STATES, False)
        delay(2, DELAY)
        os.system('cls')
    elif S[0] < 0 or S[0] > 4:
        interaction = '第%s回合: 掉出世界%s，总步数=%s' \
                      % (episode + 1, list(map(lambda x: x + 1, S)), step_counter)
        print('\r{}'.format(interaction), end='')
        delay(2, DELAY)
        os.system('cls')
    elif S[1] < 0 or S[1] > 4:
        interaction = '第%s回合: 掉出世界%s，总步数=%s' \
                      % (episode + 1, list(map(lambda x: x + 1, S)), step_counter)
        print('\r{}'.format(interaction), end='')
        delay(2, DELAY)
        os.system('cls')
    else:
        if SHOW_STEP:
            print('第{}次移动'.format(step_counter))
            env_list[S[0]][S[1]] = '■'
            print_env(env_list, N_STATES, N_STATES, False)
            delay(0.5, DELAY)
            os.system('cls')
        else:
            pass


def get_env_feedback(S, A):
    S_ = copy.deepcopy(S)
    R = 0
    if A == 'up':
        S_[0] += -1
    elif A == 'left':
        S_[1] += -1
    elif A == 'down':
        S_[0] += 1
    else:
        S_[1] += 1
    for hole in holes:
        if S_ == hole:
            R = -1
    if S_[0] < 0 or S_[0] > 4:
        R = -1
    if S_[1] < 0 or S_[1] > 4:
        R = -1
    if S == reward:
        R = 1
    return S_, R


# 在某个 state 地点, 选择行为
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
        #action_name = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
    else:
        #action_name = state_actions.idxmax()  # 贪婪模式
        action_name = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
    return action_name


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):  # 回合
        step_counter = 0
        S = [0, 0]  # 回合初始位置
        is_terminated = False  # 是否回合结束
        update_env(S, episode, step_counter)  # 环境更新
        while not is_terminated:
            S_ind = S[0] * N_STATES + S[1]
            A = choose_action(S_ind, q_table)  # 选行为
            A_ind = ACTIONS.index(A)
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            S__ind = S_[0] * N_STATES + S_[1]
            q_predict = q_table.iloc[S_ind, A_ind]  # 估算的(状态-行为)值
            if S_ != reward and holes.count(S_) == 0 and 0 <= S_[0] <= N_STATES - 1 and 0 <= S_[1] <= N_STATES - 1:
                q_target = R + LAMBDA * q_table.iloc[S__ind, :].max()  # 实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R  # 实际的(状态-行为)值 (回合结束)
                is_terminated = True  # 终止这个回合

            q_table.iloc[S_ind, A_ind] += ALPHA * (q_target - q_predict)  # q_table 更新
            S = S_  # 探索者移动到下一个 state
            update_env(S, episode, step_counter + 1)  # 环境更新
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
