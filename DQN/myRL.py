import random

import numpy as np
import tensorflow as tf

from DeepQNetwork.maze_env import Maze


class DeepQNetwork:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.acc = []
        self.loss = []

    def create_model(self):
        """创建一个隐藏层为10的神经网络"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, input_dim=self.n_features, activation='relu', name='l1'),
            tf.keras.layers.Dense(self.n_actions, activation="linear", name='l2')
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=["acc"])
        return model

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.model.predict(np.array([observation])[0])
            action = np.argmax(actions_value)
        else:
            actions_value = self.model.predict(np.array([observation])[0])
            action_ = np.argmax(actions_value)
            action = np.random.randint(0, self.n_actions)
            while action_ == action:
                action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 每 update_freq 步，将 model 的权重赋值给 target_model
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.model.get_weights())
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        replay_batch = self.memory[sample_index, :]
        s_batch = replay_batch[:, :self.n_features]
        next_s_batch = replay_batch[:, -self.n_features:]

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = replay_batch[:, self.n_features].astype(int)
        reward = replay_batch[:, self.n_features + 1]

        Q[batch_index, eval_act_index] = reward + self.gamma * np.max(Q_next, axis=1)
        # Q[batch_index, eval_act_index] = (1 - self.lr) * Q[batch_index, eval_act_index] + self.lr * (
        #       reward + self.gamma * np.max(Q_next, axis=1))

        # for i, replay in enumerate(replay_batch):
        #     _, _, a, reward, _, _ = replay
        #     a=int(a)
        #     Q[i][a] = reward + self.gamma * np.amax(Q_next[i])
        #     #Q[i][a] = reward + self.gamma * Q_next[i][a]
        #     # Q[i][a] = (1 - self.lr) * Q[i][a] + self.lr * (reward + self.gamma * np.amax(Q_next[i]))

        # 传入网络进行训练
        history = self.model.fit(s_batch, Q, verbose=0)

        self.acc.append(history.history['acc'])
        self.loss.append(history.history['loss'])

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
