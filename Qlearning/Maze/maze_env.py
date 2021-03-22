import copy
import os
import time


class Maze:
    def __init__(self, maze_size, holes, reward, birth):
        self.actions = ['up', 'left', 'down', 'right']
        self.maze_size = maze_size
        self.holes = holes
        self.reward = reward
        self.birth = birth
        self.env_list = [['□' for _ in range(self.maze_size)] for _ in range(self.maze_size)]

    def out_of_maze(self, point):
        max_ = max(point)
        min_ = min(point)
        if min_ < 0 or max_ > self.maze_size - 1:
            return True
        else:
            return False

    def print_env(self):
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                print(self.env_list[i][j], end='')
            print()

    def update_maze(self, point):
        self.env_list = [['□' for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        for hole in self.holes:
            self.env_list[hole[0]][hole[1]] = '●'
        self.env_list[self.reward[0]][self.reward[1]] = '★'
        if point!=[-1,-1]:
            self.env_list[point[0]][point[1]] = '■'
        os.system('cls')
        self.print_env()
        time.sleep(0.04)

    def env_feedback(self, point, action):
        point=copy.deepcopy(point)
        if action == 'up':
            point[0] += -1
        elif action == 'left':
            point[1] += -1
        elif action == 'down':
            point[0] += 1
        else:
            point[1] += 1

        if self.out_of_maze(point):
            R = -1
            point = [-1,-1]
            done = True
        elif self.holes.count(point) == 1:
            R = -1
            done = True
        elif self.reward == point:
            R = 1
            done = True
        else:
            R = 0
            done = False
        return point, R, done
