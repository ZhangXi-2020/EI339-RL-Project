import numpy as np
import time
import random


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
