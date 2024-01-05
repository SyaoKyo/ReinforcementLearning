import copy

import numpy as np
import random


class WOA(object):
    def __init__(self, init_X, size=30, sum_iter=1000, space=(-100, 100)):
        self.size = size  # population number
        self.iter = sum_iter  # iter number
        self.spaceUP = space[1]
        self.spaceLOW = space[0]
        self.D = np.array(init_X).shape

        self.init_X = init_X  # space sovle
        self.Lb = self.spaceLOW * np.ones_like(self.init_X)
        self.Ub = self.spaceUP * np.ones_like(self.init_X)

        self.fitness = np.zeros(self.size)  # 个体适应度
        self.best = np.zeros_like(self.init_X)  # 最好的solution
        self.fmin = 0.0

    def init_solve(self):
        '''
        初始化网络参数
        '''
        self.X = []
        for i in range(self.size):
            if i == 0:
                self.X.append(self.init_X)
            else:
                self.X.append(
                    self.init_X + self.Lb + (self.Ub - self.Lb) * np.random.uniform(0, 1, np.array(self.init_X).shape))
            self.fitness[i] = 0

        self.fmin = np.min(self.fitness)
        fmin_arg = np.argmin(self.fitness)
        self.best = self.X[fmin_arg]

    def limit(self, before_X):
        '''
        限制网络参数范围
        '''
        for i in range(self.D):
            if before_X[i] < self.spaceLOW:
                before_X[i] = self.spaceLOW
            elif before_X[i] > self.spaceUP:
                before_X[i] = self.spaceUP
        return before_X

    def run(self, env, agent, store_memory):
        '''
        运行WOA
        '''
        self.init_solve()
        seed = np.random.randint(0, 10000)
        for step in range(self.iter):
            print('step:{}'.format(step + 1))
            a = 2 - step * (2 / self.iter)  # 线性下降权重2 - 0
            # a = 1.5 - 1.2/(1+np.exp(-20*(2*step-self.iter)/(2*self.iter)))   # 改进
            a2 = -1 + step * (-1 / self.iter)  # 线性下降权重-1 - -2
            for i in range(self.size):
                # 包围
                r1 = np.random.uniform(0, 1, self.D)
                r2 = np.random.uniform(0, 1, self.D)
                A = 2 * a * r1 - a
                C = 2 * r2
                # 气泡网
                b = 1
                l = (a2 - 1) * random.random() + 1  # 原文为[-1,1]的随机数
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    # 气泡网
                    S = self.best + abs(self.best - self.X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l)
                else:
                    # 包围
                    if abs(np.random.choice(A, 1)[0]) < 1:  # 对应于 |A|<1 原论文A为向量，计算A的模则全部小于1，因此改为随机选择的方式。
                        S = self.best - abs(C * self.best - self.X[i]) * A
                    # 随机
                    else:
                        temp = np.random.randint(0, self.size - 1)
                        S = self.X[temp] - abs(C * self.X[temp] - self.X[i]) * A
                # S = self.limit(S)
                #
                agent.estimation_model.set_weights(self.X[i])
                env.seed(seed + step)
                episode_reward = 0
                for ep in range(5):
                    state = env.reset()
                    while True:
                        action = agent.choose_best_action(state)  # 预测动作，只选最优动作
                        # obs, reward, done, _ = env.step(action)
                        # episode_reward += reward
                        # store_memory(obs, action, myreward, next_state, done)
                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        x, x_dot, theta, theta_dot = next_state
                        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                        myreward = r1 + r2
                        store_memory(state, action, myreward, next_state, done)
                        state = next_state
                        env.render()
                        if done:
                            break
                Fnew = -episode_reward / 5
                if Fnew == -200.0:
                    print('第{}种群最小：{}'.format(i, -Fnew))
                    return self.X[i], Fnew
                if Fnew < self.fitness[i]:
                    self.X[i] = S
                    self.fitness[i] = Fnew

            # if self.fmin > np.min(self.fitness):
            #     print(-np.min(self.fitness))
            self.fmin = np.min(self.fitness)
            fmin_arg = np.argmin(self.fitness)
            self.best = copy.deepcopy(self.X[fmin_arg])
            print('第{}种群最小：{}'.format(fmin_arg + 1, -self.fmin))
            if self.fmin == -200.0:
                return self.best, -self.fmin
        return self.best, -self.fmin
