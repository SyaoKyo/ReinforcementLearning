import copy
import datetime
import os
from multiprocessing import Pool

import gym
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from base.agent import DQNAgent
import tensorflow as tf
from base.woa import WOA

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
# # 使用CPU训练模型
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Config:
    '''超参数
    '''

    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DQN'  # 算法名称
        self.env_name = 'CartPole-v1'  # 环境名称
        self.point_nums = 4  # 插值点总量
        self.seed = 321  # 随机种子，置0则不设置随机种子
        self.train_eps = 1000  # 训练的回合数
        self.test_eps = 10  # 测试的回合数
        self.train_step = 1000  # 训练总步数
        self.global_step = 0  # 全局步数
        self.update_step = 5  # agent学习频率
        self.woa_weights = 0.
        ################################################################################

        ################################## DQN模式 ###################################
        self.is_ddqn = True
        self.is_dueling = False
        self.is_prioritized = True
        self.is_noisy = True
        self.is_multistep = False
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.epsilon_start = 0.01  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 20000  # e-greedy策略中epsilon的衰减率
        self.lr = 0.005  # 学习率
        self.memory_capacity = int(500000)  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 4  # 目标网络的更新频率
        self.hidden_dims = 128  # 网络隐藏层
        self.num_layers = 3
        self.sigma_init = 0.3
        self.N_step = 3
        ################################################################################

        ################################# 保存结果相关参数 ################################
        self.result_path = "./data/" + self.env_name + '/results/'  # 保存结果的路径
        self.model_path = "./data/" + self.env_name + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_results(rewards, ma_rewards, eval_rewards, tag='train', path='./results'):
    ''' 保存奖励
    '''
    past_rewards = []
    past_ma_rewards = []
    past_eval_rewards = []
    if os.path.exists(path + '{}_rewards.npy'.format(tag)):
        past_rewards = np.load(path + '{}_rewards.npy'.format(tag)).tolist()
    if os.path.exists(path + '{}_ma_rewards.npy'.format(tag)):
        past_ma_rewards = np.load(path + '{}_ma_rewards.npy'.format(tag)).tolist()
    if os.path.exists(path + 'eval_rewards.npy'):
        past_eval_rewards = np.load(path + 'eval_rewards.npy').tolist()

    rewards = past_rewards + rewards
    ma_rewards = past_ma_rewards + ma_rewards
    eval_rewards = past_eval_rewards + eval_rewards
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    np.save(path + 'eval_rewards.npy'.format(tag), eval_rewards)
    print('结果保存完毕！')


def plot_rewards(rewards, ma_rewards, eval_rewards, plot_cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve of {} for {}".format(plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path + "{}_rewards_curve.png".format(tag))
    plt.show()
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve of {} for {}".format(plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(eval_rewards, label='eval rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path + "test_rewards_curve.png".format(tag))
    plt.show()


def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    n_actions = env.action_space.n  # 动作维度
    print(f"n states: {n_states}, n actions: {n_actions}")
    agent = DQNAgent(n_actions, n_states, cfg)  # 创建智能体
    if cfg.seed != 0:  # 设置随机种子
        env.seed(cfg.seed)
        np.random.seed(cfg.seed)
    if os.path.exists(cfg.model_path + 'checkpoint'):
        agent.load_weights(cfg.model_path)
    # print(list(agent.estimation_model.get_weights()))
    # exit()
    return env, agent


def run_episode(env, agent, cfg, render=True):
    total_reward = 0
    state = env.reset()
    while True:
        cfg.global_step += 1
        action = agent.choose_action(state)
        # # 初始reward
        # next_state, reward, done, _ = env.step(action)
        # agent.store_transition(state, action, reward, next_state, done)
        # 优化reward
        next_state, reward, done, _ = env.step(action)
        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        myreward = r1 + r2
        agent.store_transition(state, action, myreward, next_state, done)
        state = next_state
        if cfg.global_step % cfg.update_step == 0:
            # print(cfg.global_step)
            agent.learn()

        total_reward += reward
        if render:
            env.render()
        if done:
            break

    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, seed, render=True):
    eval_reward = []
    for i in range(5):
        env.seed(seed + i)
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.choose_best_action(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f'Test：{i + 1}/5, Reward:{episode_reward:.2f}, Epislon:{agent.epsilon(agent.frame_idx):.3f}')
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def evaluate_target(env, agent, seed, render=True):
    eval_reward = []
    for i in range(5):
        env.seed(seed + i)
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.choose_best_action_by_target(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f'Target Test：{i + 1}/5, Reward:{episode_reward:.2f}, Epislon:{agent.epsilon(agent.frame_idx):.3f}')
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def train(cfg, env, agent):
    ''' 训练
    '''
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    eval_rewards = []
    i_ep = 0
    # 训练50次验证5次
    while i_ep < cfg.train_eps:
        # 训练
        for _i_train in range(25):
            i_ep += 1
            ep_reward = run_episode(env, agent, cfg)  # 记录一回合内的奖励
            # print(ep_reward)
            rewards.append(ep_reward)
            # print(ma_rewards)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            if (i_ep) % 1 == 0:
                print(
                    f'Episode：{i_ep}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Epislon:{agent.epsilon(agent.frame_idx):.3f}')
            if i_ep >= cfg.train_eps:
                break
        # 测试
        random_seed = np.random.randint(0, 10000)
        new_env = gym.make(cfg.env_name)
        eval_reward = evaluate(new_env, agent, random_seed)
        # target_eval_reward = evaluate_target(new_env, agent, random_seed)
        # if eval_reward >= target_eval_reward:
        #     print('Q:{}\tQ_t:{}'.format(eval_reward,target_eval_reward))
        #     print('update target model.')
        #     agent.update_target_model()
        eval_rewards.append(eval_reward)
        print(f'Test reward:{eval_reward:.2f}')

    print('完成训练！')
    env.close()
    return rewards, ma_rewards, eval_rewards


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            env.render()
            action = agent.choose_action(state, training=False)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    env.close()
    return rewards, ma_rewards


def create_env_and_train(cfg):
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards, eval_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save_weights(cfg.model_path, overwrite=True)  # 保存模型
    save_results(rewards, ma_rewards, eval_rewards, tag='train',
                 path=cfg.result_path)  # 保存结果
    # plot_rewards(rewards, ma_rewards, eval_rewards, cfg, tag="train")  # 画出结果


if __name__ == "__main__":
    cfg = Config()
    pool = Pool(processes=os.cpu_count())
    for i in range(32):
        cfg.is_ddqn = bool(i & 16)
        cfg.is_dueling = bool(i & 8)
        cfg.is_prioritized = bool(i & 4)
        cfg.is_noisy = bool(i & 2)
        cfg.is_multistep = bool(i & 1)
        cfg.result_path = "./data/" + cfg.env_name + '/results/' + str(i) + '/'
        cfg.model_path = "./data/" + cfg.env_name + '/models/' + str(i) + '/'
        cfg_temp = copy.deepcopy(cfg)
        # 训练
        pool.apply_async(func=create_env_and_train, args=(cfg_temp,))
    pool.close()
    pool.join()
    exit()

    # 训练
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards, eval_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save_weights(cfg.model_path, overwrite=True)  # 保存模型
    save_results(rewards, ma_rewards, eval_rewards, tag='train',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, eval_rewards, cfg, tag="train")  # 画出结果
    # 测试
    # agent.load_weights(eval_cfg.model_path)  # 导入模型
    # rewards, ma_rewards = test(eval_cfg, env, agent)
    # save_results(rewards, ma_rewards, tag='test',
    #              path=cfg.result_path)  # 保存结果
    # plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果
