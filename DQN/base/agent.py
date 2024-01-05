import time
import gym
import tensorflow as tf
import numpy as np
from base.model import MyDQN
import math
from base.memory import ReplayBuffer, PrioritizedReplayBuffer
import random
from base.woa import WOA


# # 设置 GPU 显存使用方式为：为增长式占用-----------------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:  # 设置 GPU 为增长式占用
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         # 打印异常
#         print(e)


class DQNAgent():
    def __init__(self, n_actions, n_features, cfg):
        tf.random.set_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.n_features = n_features
        self.n_actions = n_actions
        self.cfg = cfg
        self.gamma = self.cfg.gamma
        self.target_update = self.cfg.target_update
        self.count_step = 0
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.is_ddqn = self.cfg.is_ddqn
        self.is_dueling = self.cfg.is_dueling
        self.is_prioritized = self.cfg.is_prioritized
        self.is_noisy = self.cfg.is_noisy
        self.is_multistep = self.cfg.is_multistep
        self.N_step = self.cfg.N_step
        self.woa_weights = self.cfg.woa_weights

        # create net
        self.target_model = self._build_net()
        self.estimation_model = self._build_net()

        inputs = tf.keras.layers.Input(shape=(n_features))
        self.target_model(inputs, training=False)
        self.estimation_model(inputs, training=False)

        if self.is_prioritized:
            self.memory = PrioritizedReplayBuffer(self.cfg.memory_capacity)  # 经验回放
        else:
            self.memory = ReplayBuffer(self.cfg.memory_capacity)  # 经验回放

        if self.is_multistep:
            self.N_step_buffer = []
            self.N_step_idx = 0

        self.loss_meters = []
        self.mae_meters = []

    def _build_net(self):
        model = MyDQN(self.n_actions, self.n_features, self.cfg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr)
        # optimizer = tf.keras.optimizers.RMSprop(lr=self.cfg.lr)
        # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', 'mae'])
        model.optimizer = optimizer
        # model.build(input_shape=[None, self.n_features])
        return model

    def update_target_model(self):
        print('target update')
        # # 软更新
        # weights = .5
        # target = self.target_model.get_weights()
        # estimation = self.estimation_model.get_weights()
        # for idx in range(len(self.target_model.get_weights())):
        #     target[idx] *= (1 - weights)
        #     estimation[idx] *= weights
        #     target[idx] = target[idx] + estimation[idx]
        # self.target_model.set_weights(target)

        # 硬更新
        self.target_model.set_weights(self.estimation_model.get_weights())

    def woa_update(self):
        woa = WOA(self.estimation_model.get_weights(), size=25, sum_iter=10, space=(-.1, .1))
        woa_env = gym.make('CartPole-v0')
        woa_agent = DQNAgent(woa_env.action_space.n, woa_env.observation_space.shape[0], self.cfg)  # 创建智能体
        best_X, _ = woa.run(woa_env, woa_agent, self.store_transition)
        estimation = self.estimation_model.get_weights()
        for idx in range(len(self.target_model.get_weights())):
            estimation[idx] = self.woa_weights * best_X[idx] + (1 - self.woa_weights) * estimation[idx]
        self.estimation_model.set_weights(estimation)

    def choose_action(self, state, training=False):

        state = state[np.newaxis, :]
        if self.cfg.is_noisy:
            actions_value = self.estimation_model.predict(state, verbose=0)
            action = np.argmax(actions_value)
        else:
            # actions_value = self.estimation_model(np.array([state])[0], training=training)
            # print(state.shape)
            actions_value = self.estimation_model.predict(state, verbose=0)
            # print(actions_value.shape)
            # exit()
            action = np.argmax(actions_value)
            if random.random() < self.epsilon(self.frame_idx):
                action_ = np.random.randint(0, self.n_actions)
                while action_ == action:
                    action_ = np.random.randint(0, self.n_actions)
                action = action_
        # print(actions_value, action)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        if self.is_prioritized:
            if self.is_multistep:
                # 如果没有达到设定的步数，return
                if len(self.N_step_buffer) < self.N_step:
                    # 把当前经验放入N_step buffer中
                    self.N_step_buffer.append((state, action, reward, next_state, done))
                    return

                #  计算N步奖励
                R = 0
                idx = self.N_step_idx
                flag = True
                while idx != self.N_step_idx or flag:
                    flag = False
                    # print(idx, self.N_step_idx, self.N_step, self.N_step_buffer)
                    # 若该步不是终止步，则累加对应的衰减奖励
                    if self.N_step_buffer[idx % self.N_step][4] is False:
                        R += self.N_step_buffer[idx % self.N_step][2] * (
                                self.gamma ** ((idx - self.N_step_idx) % self.N_step))
                        idx = (idx + 1) % self.N_step
                    # 若该步是终止步，则直接累加最终步骤奖励至满足n步的奖励
                    else:
                        R += self.N_step_buffer[idx % self.N_step][2] * ((idx - self.N_step_idx) % self.N_step)
                        break
                # 存入经验回放池
                save_state, save_action, save_reward, save_next_state, save_done = self.N_step_buffer[self.N_step_idx]
                self.memory.push((save_state, save_action, save_reward, save_next_state, save_done))
                # 用最新的经验替换掉存在多步缓存中最老的经验
                self.N_step_buffer[self.N_step_idx] = (state, action, reward, next_state, done)
                self.N_step_idx = (self.N_step_idx + 1) % self.N_step
            else:
                self.memory.push((state, action, reward, next_state, done))

        else:
            if self.is_multistep:
                # 如果没有达到设定的步数，return
                if len(self.N_step_buffer) < self.N_step:
                    # 把当前经验放入N_step buffer中
                    self.N_step_buffer.append((state, action, reward, next_state, done))
                    return

                #  计算N步奖励
                R = 0
                idx = self.N_step_idx
                flag = True
                while idx != self.N_step_idx or flag:
                    flag = False
                    # 若该步不是终止步，则累加对应的衰减奖励
                    if self.N_step_buffer[idx % self.N_step][4] is False:
                        R += self.N_step_buffer[idx % self.N_step][2] * (
                                self.gamma ** ((idx - self.N_step_idx) % self.N_step))
                        idx = (idx + 1) % self.N_step
                    # 若该步是终止步，则直接累加最终步骤奖励至满足n步的奖励
                    else:
                        R += self.N_step_buffer[idx % self.N_step][2] * ((idx - self.N_step_idx) % self.N_step)
                        break
                # 存入经验回放池
                save_state, save_action, save_reward, save_next_state, save_done = self.N_step_buffer[self.N_step_idx]
                self.memory.push(save_state, save_action, save_reward, save_next_state, save_done)
                # 用最新的经验替换掉存在多步缓存中最老的经验
                self.N_step_buffer[self.N_step_idx] = (state, action, reward, next_state, done)
                self.N_step_idx = (self.N_step_idx + 1) % self.N_step
            else:
                self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        if self.count_step % self.target_update == 0:
            self.update_target_model()
            if self.count_step != 0 and self.count_step % (
                    5 * self.target_update) == 0 and self.cfg.woa_weights != 0.0:
                self.woa_update()
        self.count_step += 1
        self.frame_idx += 1
        ISWeights = []
        if self.is_prioritized:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, batch_index, ISWeights = self.memory.sample(
                self.batch_size)
            state_batch = np.array(state_batch)  # [batch_size, 4]
            action_batch = np.array(action_batch)  # [batch_size,]
            reward_batch = np.array(reward_batch)  # [batch_size,]
            next_state_batch = np.array(next_state_batch)  # [batch_size, 4]
            done_batch = np.float32(done_batch)  # [batch_size,]
            mem_index = np.array(batch_index)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            # batch_index =np.arange(self.batch_size, dtype=np.int32)
        else:
            # 从经验回放中(replay base)中随机采样一个批量的转移(transition)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

            state_batch = np.array(state_batch)  # [batch_size, 4]
            action_batch = np.array(action_batch)  # [batch_size,]
            reward_batch = np.array(reward_batch)  # [batch_size,]
            next_state_batch = np.array(next_state_batch)  # [batch_size, 4]
            done_batch = np.float32(done_batch)  # [batch_size,]
            batch_index = np.arange(self.batch_size, dtype=np.int32)
        if not ISWeights:
            ISWeights = np.array([1.0] * self.batch_size)
        if self.is_ddqn:
            q_values = self.estimation_model.predict(state_batch, verbose=0)
            q_next_value = self.estimation_model.predict(next_state_batch, verbose=0)
            next_q_values = self.target_model.predict(next_state_batch, verbose=0)  # 计算下一时刻的状态(s_t_,a)对应的Q值
            q_next_value = tf.argmax(q_next_value, axis=1)
            next_q_values = next_q_values[batch_index, q_next_value]
            # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
            # expected_q_values = q_values.copy()
            # expected_q_values[batch_index, action_batch] = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            # y_old = expected_q_values[batch_index, action_batch]
            # y = q_values[batch_index, action_batch]
            target_value = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            y_old = target_value
            y = tf.reduce_max(q_values, axis=1)
        else:
            q_values = self.estimation_model.predict(state_batch, verbose=0)
            next_q_values = self.target_model.predict(next_state_batch, verbose=0)  # 计算下一时刻的状态(s_t_,a)对应的Q值
            next_q_values = tf.reduce_max(next_q_values, axis=1)
            # print('best', next_q_values.shape)
            # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
            # expected_q_values = q_values.copy()
            # expected_q_values[batch_index, action_batch] = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            # y_old = expected_q_values[batch_index, action_batch]
            # y = q_values[batch_index, action_batch]

            target_value = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            y_old = target_value
            y = tf.reduce_max(q_values, axis=1)

        # 计算当前状态(s_t,a)对应的Q(s_t, a)并训练
        # history = self.estimation_model.fit(state_batch, target_value, epochs=1, verbose=0)
        # print(target_value, target_value.shape)
        self._train_model(action_batch, state_batch, target_value, ISWeights, epochs=1)
        if self.is_prioritized:
            diff = np.abs(y_old - y)
            self.memory.batch_update(mem_index, diff)

        if self.is_noisy:
            self.estimation_model.reset_noise()
            self.target_model.reset_noise()

        # self.loss_meters.append(np.mean(history.history['loss']))
        # self.mae_meters.append(np.mean(history.history['mae']))

    def save_weights(self, filepath, overwrite=False):
        self.estimation_model.save_weights(filepath + 'weights.h5f', overwrite=overwrite)
        # self.target_model.save_weights(filepath + 'weights.h5f', overwrite=overwrite)

    def load_weights(self, filepath):
        self.estimation_model.load_weights(filepath + 'weights.h5f')
        self.target_model.set_weights(self.estimation_model.get_weights())
        # self.target_model.load_weights(filepath + 'weights.h5f')
        # self.estimation_model.set_weights(self.target_model.get_weights())

    def _train_step(self, action, features, labels, weights=1.0):
        """ 训练步骤
        """
        # print(features, features.shape)
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            predictions = self.estimation_model(features, training=True)
            # enum_action = np.array(list(enumerate(action))).astype(np.int32)
            enum_action = list(enumerate(action))
            # print(enum_action)
            pred_action_value = tf.gather_nd(predictions, indices=enum_action)
            # 带权重的mae
            # loss = weights * tf.math.abs(labels - pred_action_value)
            # 带权重的mse
            # 整体（标量）
            loss = tf.losses.MeanSquaredError()(labels, pred_action_value, sample_weight=[weights])
            # print(loss)
            # loss = tf.reduce_mean(weights * tf.pow((labels - pred_action_value), 2))
            # 单个（张量）
            # loss = tf.reduce_mean([weights] * tf.expand_dims(tf.pow((labels - pred_action_value), 2), axis=0), axis=0)
            # print(loss)
            # huber
            # loss = tf.keras.losses.huber(labels, pred_action_value)
            # print(loss)
            # error_abs = tf.math.abs(labels - pred_action_value)
            # delta = 1.0
            # residual = weights * tf.abs(pred_action_value - labels)
            # condition = [tf.less(residual, delta), tf.greater_equal(residual, delta)]
            # small_res = 0.5 * tf.square(residual)
            # large_res = delta * residual - 0.5 * tf.square(delta)
            # loss = tf.experimental.numpy.select(condition, [small_res, large_res])
        gradients = tape.gradient(loss, self.estimation_model.trainable_variables)
        self.estimation_model.optimizer.apply_gradients(zip(gradients, self.estimation_model.trainable_variables))
        # self.estimation_model.train_loss.update_state(loss)

    def _train_model(self, action, features, labels, weights, epochs=1):
        """
        训练模型
        """
        for _ in tf.range(1, epochs + 1):
            self._train_step(action, features, labels, weights)

    def choose_best_action(self, state):
        '''
        选择最佳动作
        '''
        state = state[np.newaxis, :]
        actions_value = self.estimation_model.predict(state, verbose=0)
        action = np.argmax(actions_value)
        return action

    def choose_best_action_by_target(self, state):
        '''
        选择最佳动作
        '''
        state = state[np.newaxis, :]
        actions_value = self.target_model.predict(state, verbose=0)
        action = np.argmax(actions_value)
        return action
