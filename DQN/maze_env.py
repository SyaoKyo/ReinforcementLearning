import numpy as np
import time
import tkinter as tk

UNIT = 40  # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 1, UNIT * 0])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')

        hell2_center = origin + np.array([UNIT * 4, UNIT * 1])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        hell3_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')

        hell4_center = origin + np.array([UNIT * 0, UNIT * 3])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')

        hell5_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')

        hell6_center = origin + np.array([UNIT * 2, UNIT * 4])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + np.array([UNIT * 3, UNIT * 4])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return np.array([(self.canvas.coords(self.rect)[1] - 5) / UNIT, (self.canvas.coords(self.rect)[0] - 5) / UNIT])

    def _is_border(self, x, y, action):
        if x == 0:
            if y == 0:
                if action == 0 or action == 2:
                    return True
                else:
                    return False
            elif y == MAZE_W - 1:
                if action == 0 or action == 3:
                    return True
                else:
                    return False
            elif action == 0:
                return True
            else:
                return False
        elif x == MAZE_H - 1:
            if y == 0:
                if action == 1 or action == 2:
                    return True
                else:
                    return False
            elif y == MAZE_W - 1:
                if action == 1 or action == 3:
                    return True
                else:
                    return False
            elif action == 1:
                return True
            else:
                return False
        elif y == 0:
            if x == 0:
                if action == 0 or action == 2:
                    return True
                else:
                    return False
            elif x == MAZE_H - 1:
                if action == 1 or action == 2:
                    return True
                else:
                    return False
            elif action == 2:
                return True
            else:
                return False
        elif y == MAZE_H - 1:
            if x == 0:
                if action == 0 or action == 3:
                    return True
                else:
                    return False
            elif x == MAZE_H - 1:
                if action == 1 or action == 3:
                    return True
                else:
                    return False
            elif action == 3:
                return True
            else:
                return False
        else:
            return False

    def step(self, action):
        s = self.canvas.coords(self.rect)
        x = (s[1] - 5) / UNIT
        y = (s[0] - 5) / UNIT
        if self._is_border(x, y, action):
            return np.array([None, None]), -1, False, False
        else:
            base_action = np.array([0, 0])
            if action == 0:  # up
                if s[1] > UNIT:
                    base_action[1] -= UNIT
            elif action == 1:  # down
                if s[1] < (MAZE_H - 1) * UNIT:
                    base_action[1] += UNIT
            elif action == 3:  # right
                if s[0] < (MAZE_W - 1) * UNIT:
                    base_action[0] += UNIT
            elif action == 2:  # left
                if s[0] > UNIT:
                    base_action[0] -= UNIT

            self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

            next_coords = self.canvas.coords(self.rect)  # next state

            # reward function
            if next_coords == self.canvas.coords(self.oval):
                reward = 1
                done = True
            elif next_coords in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2),
                                 self.canvas.coords(self.hell3), self.canvas.coords(self.hell4),
                                 self.canvas.coords(self.hell5),
                                 self.canvas.coords(self.hell6)
                                 ]:
                reward = -1
                done = True
            else:
                reward = 0
                done = False
            s_ = np.array([(next_coords[1] - 5) / UNIT, (next_coords[0] - 5) / UNIT])
            return s_, reward, done, True

    def render(self):
        # time.sleep(0.01)
        self.update()
