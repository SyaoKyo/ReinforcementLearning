import copy
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import gym
import numpy as np


class ActorCritic(tf.keras.Model):

    def __init__(self, action_dim, hidden_dim, num_layers):
        super(ActorCritic, self).__init__()
        # 演员网络
        self.actor_fc_layers = [tf.keras.layers.Dense(hidden_dim, activation='relu') for _ in
                                range(num_layers)]
        self.action_layer = tf.keras.layers.Dense(action_dim, activation='softmax')
        # 评论家网络
        self.critic_fc_layers = [tf.keras.layers.Dense(hidden_dim, activation='relu') for _ in
                                 range(num_layers)]
        self.critic_layer = tf.keras.layers.Dense(1)

    def call(self, x, training=None, mask=None):
        actor_x = copy.deepcopy(x)
        for i, layer in enumerate(self.actor_fc_layers):
            actor_x = layer(actor_x)
        logits_p = self.action_layer(actor_x)

        critic_x = copy.deepcopy(x)
        for i, layer in enumerate(self.actor_fc_layers):
            critic_x = layer(critic_x)
        value = self.critic_layer(critic_x)
        return logits_p, value


class A2C:
    def __init__(self, cfg):
        self.model = ActorCritic(cfg.action_dim, cfg.hidden_dim, cfg.num_layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
        self.gamma = cfg.gamma
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.entropy_coef = cfg.entropy_coef
        self.critic_loss_coef = cfg.critic_loss_coef

    def compute_loss(self, actions, logits, values, advantages):
        actions_one_hot = tf.one_hot(actions, self.action_dim)
        action_prob = tf.reduce_sum(actions_one_hot * logits, axis=1)
        advantages = tf.stop_gradient(advantages)
        entropy = -tf.reduce_sum(logits * tf.math.log(logits), axis=1)

        # actor_loss = -tf.reduce_mean(tf.math.log(action_prob + 1e-10) * advantages)
        # critic_loss = tf.reduce_mean((advantages - values) ** 2)
        # total_loss = actor_loss + critic_loss * self.critic_loss_coef - entropy * self.entropy_coef
        # return total_loss

        # total_loss
        actor_loss = -tf.reduce_mean(tf.math.log(action_prob + 1e-10) * advantages + self.entropy_coef * entropy)
        critic_loss = tf.reduce_mean((advantages - values) ** 2)
        return actor_loss + critic_loss

    def train_step(self, states, actions, rewards, next_states, dones):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            next_logits, next_values = self.model(next_states)

            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values

            loss = self.compute_loss(actions, logits, values, advantages)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


class Config:
    def __init__(self) -> None:
        self.algo_name = 'A2C'  # 算法名称
        self.env_id = 'CartPole-v0'  # 环境id
        self.seed = 321  # 随机种子，便于复现，0表示不设置
        self.train_eps = 500  # 训练的总步数
        self.test_eps = 20  # 测试的总回合数
        self.n_steps = 5  # 更新策略的轨迹长度
        self.max_steps = 200  # 测试时一个回合中能走的最大步数
        self.gamma = 0.95  # 折扣因子
        self.lr = 0.0005  # 网络学习率
        self.critic_loss_coef = 0.5  # 值函数系数值
        self.entropy_coef = 0.001  # 策略熵系数值
        self.hidden_dim = 128  # 网络的隐藏层维度
        self.num_layers = 3


def env_agent_config(cfg):
    env = gym.make(cfg.env_id)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print(f"状态空间维度：{state_dim}，动作空间维度：{num_actions}")
    setattr(cfg, "state_dim", state_dim)  # 更新state_dim到cfg参数中
    setattr(cfg, "action_dim", num_actions)  # 更新action_dim到cfg参数中
    # 实例化 A2C 模型
    agent = A2C(cfg)  # 创建agent实例

    if cfg.seed != 0:  # 设置随机种子
        env.seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)  # 设置TensorFlow随机种子
        # os.environ['PYTHONHASHSEED'] = str(cfg.seed)  # 配置Python脚本的随机种子
        # 关闭CUDNN的随机性以保证实验结果的重复性
        # tf.config.experimental_run_functions_eagerly(True)
        # tf.compat.v1.disable_eager_execution()

    return env, agent


def train(env, agent, cfg):
    rewards = []  # 记录所有回合的奖励
    for episode in range(cfg.train_eps):
        state = env.reset()
        episode_reward = 0
        env.render()
        while True:
            # 在环境中采取动作并观察下一个状态和奖励
            logits, _ = agent.model(np.array([state]))
            action = np.random.choice(range(cfg.action_dim), p=np.squeeze(logits.numpy()))
            next_state, reward, done, _ = env.step(action)
            # 自定义奖励
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            myreward = r1 + r2

            # 存储经验
            agent.train_step([state], [action], [myreward], [next_state], [done])
            episode_reward += reward
            state = next_state
            env.render()
            if done:
                break
        rewards.append(episode_reward)
        # 打印每个回合的信息
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")
    env.close()  # 关闭环境
    return rewards


def smooth(data, weight=0.9):
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, cfg):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve for {}".format(cfg.env_id))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.savefig("./a2c_rewards_curve.png")
    plt.show()


if __name__ == "__main__":
    cfg = Config()

    env, agent = env_agent_config(cfg)

    rewards = train(env, agent, cfg)

    plot_rewards(rewards, cfg)
