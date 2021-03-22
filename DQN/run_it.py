from maze_env import Maze
from myRL import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


def run_maze():
    step = 0
    Qtable_index = []
    goal = 0
    fail = 0
    episode = 0
    while True:

        # initial observation
        observation = env.reset()
        Qtable = []
        if not [observation[0], observation[1]] in Qtable_index:
            Qtable_index.append([observation[0], observation[1]])
        print('第%d回合' % (episode + 1))
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, valid_step = env.step(action)
            # print([observation[1], observation[0]], [observation_[1], observation_[0]])
            if valid_step:
                print('%s->' % env.action_space[action], end='')
            # save in store
            RL.store_transition(observation, action, reward, observation_)
            if not [observation[0], observation[1]] in Qtable_index:
                Qtable_index.append([observation[0], observation[1]])
                Qtable_index = sorted(Qtable_index, key=itemgetter(0, 1))

            if (step > 1000) and (step % 5 == 0):
                RL.learn()

            # swap observation
            if valid_step:
                observation = observation_

            # break while loop when end of this episode
            if done:
                print('end')
                if reward < 0:
                    fail += 1
                if reward > 0:
                    goal += 1
                    print('goal! totally：%d' % goal)
                    for index in Qtable_index:
                        index = np.array(index)[np.newaxis, :]
                        Qtable.append(RL.model.predict(index))
                    print('Postion\tUp\t\tDown\tLeft\tRight')
                    for i in range(len(Qtable_index)):
                        print('[%d, %d]\t%.4f\t%.4f\t%.4f\t%.4f' % (
                            Qtable_index[i][0], Qtable_index[i][1], Qtable[i][0][0], Qtable[i][0][1], Qtable[i][0][2],
                            Qtable[i][0][3]))
                break
            step += 1
        if goal >= fail/10:
            break
        episode += 1
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=1024,
                      memory_size=1024,
                      batch_size=128
                      )
    env.after(100, run_maze)
    env.mainloop()

    acc = RL.acc
    loss = RL.loss

    epochs = range(len(acc))  # Get number of epochs

    # 画accuracy曲线和loss曲线
    plt.plot(epochs, loss, 'b')
    plt.xlabel("Epochs")
    plt.legend(["Loss"])

    # plt.figure()
    plt.show()

    # RL.plot_cost()
