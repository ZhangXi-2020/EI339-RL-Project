import numpy as np
import time
import random
import matplotlib.pyplot as plt
import plotly.figure_factory as FF
from scipy.spatial import Delaunay
import seaborn as sns
import numpy as np
from matplotlib import cm

class State():
    def __init__(self, dealer_card, player_sum):
        self.dealer_card = dealer_card
        self.player_sum = player_sum


class Game():
    def __init__(self):
        self.action_space = ['hit', 'stick']
        self.n_actions = len(self.action_space)
        self.player_score = random.randint(1, 10)
        self.dealer_score = random.randint(1, 10)
        self.dealer_card = self.dealer_score

    def reset(self):
        self.player_score = random.randint(1, 10)
        self.dealer_score = random.randint(1, 10)
        # return State(self.dealer_score, self.player_score)
        return [self.dealer_score, self.player_score]

    def set(self, dealer_card, player_sum):
        self.player_score = player_sum
        self.dealer_score = dealer_card

    def get_state(self):
        return self.dealer_card, self.player_score

    def draw_card(self):
        type_idx = random.randint(1, 3)
        #draw a black card
        if type_idx % 3:
            black = random.randint(1, 10)
            return black
        # draw a red card
        else:
            red = random.randint(1, 10)
            return (0 - red)

    def hit_value(self,value):
        self.player_score += value
        if self.player_score > 21 or self.player_score < 1:
            return -1
        else:
            return 0

    def step(self, action):
        if action == 0:  # player hit
            score = self.draw_card()
            self.player_score += score
        
            if self.player_score > 21 or self.player_score < 1:  # player bust
                reward = -1
                state_ = 'terminal'
                return state_, reward
            
            # game continue
            reward = 0
            # state_ = State(self.dealer_score, self.player_score)
            state_ = [self.dealer_score, self.player_score]
            return state_, reward
        
        elif action == 1:  # player stick so that game over
            state_ = 'terminal'

            # dealer decide
            while True:
                if self.dealer_score < 16 and self.dealer_score >= 1:
                    score = self.draw_card()
                    self.dealer_score += score
                if self.dealer_score > 21 or self.dealer_score < 1:  # dealer bust
                    reward = 1                
                    return state_, reward
                if self.dealer_score >= 16:
                    break
            
            # compare stage
            if self.player_score > self.dealer_score:
                reward = 1
            elif self.player_score < self.dealer_score:
                reward = -1
            elif self.player_score == self.dealer_score:
                reward = 0
            else:
                raise Exception('Game Error')
            
            return state_, reward


CARD_VALUE_MAX = 10
CARD_VALUE_MIN = 1
DEALER_FIRST_CARD_SPACE = np.arange(CARD_VALUE_MIN, CARD_VALUE_MAX + 1)
class Visualizer:
    def __init__(self):
        pass
    @staticmethod
    def prepare_axises(Q):
        X, Y, Z = [], [], []
        for dealer in DEALER_FIRST_CARD_SPACE:
            for player in range(1,21 + 1):
                state = dealer, player
                X.append(dealer)
                Y.append(player)
                Z.append(max(Q[state,0], Q[state, 1]))
        return X, Y, Z
    @staticmethod
    def draw_surf(X, Y, Z, title):
        fig = plt.figure(figsize = (10, 8))
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)
        ax.set_xlabel('dealer')
        ax.set_ylabel('player')
        ax.set_zlabel('Value')
        ax.set_xticklabels(range(1, CARD_VALUE_MAX + 1))
        ax.set_yticklabels(range(1, 21 + 1))
        plt.title(title)
        plt.xticks(DEALER_FIRST_CARD_SPACE)
        plt.yticks(range(0, 21))
        plt.show()
    @staticmethod
    def draw_deluanay_surf(X, Y, Z, title):
        u = np.linspace(0, 2*np.pi, 21)
        v = np.linspace(0, 2*np.pi, CARD_VALUE_MAX)
        u,v = np.meshgrid(u,v)
        u = u.flatten()
        v = v.flatten()

        points2D = np.vstack([u,v]).T
        tri = Delaunay(points2D)
        simplices = tri.simplices

        fig = FF.create_trisurf(z=Z, x = X, y = Y, simplices=simplices)

        scene = dict(
            xaxis = dict(nticks=10, range=[CARD_VALUE_MIN, CARD_VALUE_MAX], tick0 = 1),
            yaxis = dict(nticks=21, range=[1, 21], ticks='outside', tick0 = 1),
            zaxis = dict(nticks=10, range=[np.min(Z) - 0.5, np.max(Z) + 0.5],),
            xaxis_title='Dealer',
            yaxis_title='Player',
            zaxis_title='Value',
        )
        fig.update_layout(scene = scene,
                        title = title,
                        autosize = True,
                        width=700,
                        height=500,
                        margin = dict(l=65, r=50, b=65, t=90)
        )
        fig.show()
    @staticmethod
    def visualize(Q, title):
        X, Y, Z = Visualizer.prepare_axises(Q)
        Visualizer.draw_deluanay_surf(X, Y, Z, title)
        Visualizer.draw_surf(X, Y, Z, title)

    @staticmethod
    def draw2d_square_array(array, title):
        fig, ax = plt.subplots(figsize = (11, 11))
        sns.heatmap(
            array,
            linewidths = 0.1,
            annot = True,
            xticklabels = 1,
            yticklabels = 1,
            cbar = False,
            fmt = "0.2f",
            square = True,
            ax = ax,
            alpha = 0.8,
        )
        ax.set_title(title)
        plt.show()

    @staticmethod
    def draw2d_array(array, title, low_lim = False):
        # array = np.arange(1, 32).reshape(21, 10)
        fig, ax = plt.subplots(figsize = (8, 8))
        sns.heatmap(
            array,
            linewidths = 0.1,
            annot = True,
            xticklabels = 1,
            yticklabels = 1,
            cbar = False,
            fmt = "0.2f",
            cmap = 'coolwarm',
            ax = ax,
            alpha = 0.8,
        )
        low = 1 if low_lim else 0
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xticklabels(range(low, CARD_VALUE_MAX + 1))
        ax.set_yticklabels(range(low, 21 + 1))
        ax.set_xlabel("Dealer's first card")
        ax.set_ylabel("Player's score")
        ax.set_title(title)
        plt.show()