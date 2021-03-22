# project    :don't to try
# fileName   :run_it.py
# user       :cheng
# createDate :2021-1-28 21:21
import os

from maze_env import Maze
from myRL import QLearningTable, SarsaTable, SarsaLambdaTable
import time

N_STATES = 5
HOLES = [[0, 1], [1, 4], [2, 2], [3, 0], [3, 3], [4, 2]]
REWARD = [4, 3]
BIRTH = [0, 0]
ACTIONS = ['up', 'left', 'down', 'right']
MAX_EPISODES = 100


def QL_update():
    for episode in range(MAX_EPISODES):
        # initial observation
        R = BIRTH
        env.update_maze(R)
        while True:
            # RL choose action based on observation
            A = RL.choose_action(str((R[0], R[1])))

            # RL take action and get next observation and reward
            R_, reward, done = env.env_feedback(R, A)

            # RL learn from this transition
            RL.learn(str((R[0], R[1])), A, reward, str((R_[0], R_[1])))

            # swap observation
            R = R_

            # fresh env
            env.update_maze(R)
            # break while loop when end of this episode
            if done:
                time.sleep(1)
                break


def Sarsa_update():
    for episode in range(MAX_EPISODES):
        # initial observation
        R = BIRTH
        env.update_maze(R)
        # RL choose action based on observation
        A = RL.choose_action(str((R[0], R[1])))
        while True:

            # RL take action and get next observation and reward
            R_, reward, done = env.env_feedback(R, A)

            # RL choose action based on next observation
            A_ = RL.choose_action(str((R_[0], R_[1])))

            # RL learn from this transition
            RL.learn(str((R[0], R[1])), A, reward, str((R_[0], R_[1])), A_)

            # swap observation and action
            R = R_
            A = A_

            # fresh env
            env.update_maze(R)
            # break while loop when end of this episode
            if done:
                time.sleep(1)
                break


def SarsaLambda_update():
    for episode in range(MAX_EPISODES):
        # initial observation
        R = BIRTH
        env.update_maze(R)
        # RL choose action based on observation
        A = RL.choose_action(str((R[0], R[1])))
        # initial all zero eligibility trace
        RL.eligibility_trace *= 0
        while True:

            # RL take action and get next observation and reward
            R_, reward, done = env.env_feedback(R, A)

            # RL choose action based on next observation
            A_ = RL.choose_action(str((R_[0], R_[1])))

            # RL learn from this transition
            RL.learn(str((R[0], R[1])), A, reward, str((R_[0], R_[1])), A_)

            # swap observation and action
            R = R_
            A = A_

            # fresh env
            env.update_maze(R)
            # break while loop when end of this episode
            if done:
                time.sleep(1)
                break


if __name__ == "__main__":
    env = Maze(N_STATES, HOLES, REWARD, BIRTH)
    # Q-Learning
    # RL = QLearningTable(actions=ACTIONS)
    # QL_update()

    # Sarsa
    RL = SarsaTable(actions=ACTIONS)
    Sarsa_update()

    # Sarsa(Lambda)
    # RL = SarsaLambdaTable(actions=ACTIONS)
    # SarsaLambda_update()

    os.system('cls')
    print(RL.q_table)
    os.system('pause')
