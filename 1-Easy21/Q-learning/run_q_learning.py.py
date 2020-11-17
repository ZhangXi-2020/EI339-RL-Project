from env import Game
from QLearning import QLearning
import pandas as pd


# 超参数
exp_id = 1
EPSILON = 0.8                     # 贪婪度 greedy
ALPHA = 0.01                      # 学习率
GAMMA = 1                         # 奖励递减值
MAX_EPISODES = 50000              # 最大回合数

def simulation(episode):
    win = fail = equal = 0
    for epi in range(10000):
        state = env.reset()                                     # initial state
        while True:            
            action = RL.choose_best_action(str(state))          # RL choose action based on state
            state_, reward = env.step(action)                   # RL take action and get next state and reward
            state = state_                                      # swap state
            if state == 'terminal':                             # break while loop when end of this episode
                break
        if reward == 1:
            win += 1
        elif reward == -1:
            fail += 1
        elif reward == 0:
            equal += 1
        
    print(win, equal, fail)

    lr = ALPHA
    e = EPSILON
    file_path = './result/result_exp'+ str(exp_id) + '_lr_' + str(lr) + '_e_'+ str(e) +'.txt'
    with open(file_path, 'a') as f:
        line = str(episode) + '\t' + str(win) + '\t' + str(equal) + '\t' + str(loss) + '\n'
        f.write(line)
        win = fail = equal = 0


def update(env, RL):
    for episode in range(MAX_EPISODES):
        state = env.reset()                                     # initial state
        while True:            
            action = RL.choose_action(str(state))               # RL choose action based on state
            state_, reward = env.step(action)                   # RL take action and get next state and reward
            RL.learn(str(state), action, reward, str(state_))   # RL learn from this transition
            state = state_                                      # swap state
            if state == 'terminal':                             # break while loop when end of this episode
                break
        
        if episode % 500 == 0:
            simulation(episode)
        # writer = pd.ExcelWriter('./file1.xlsx')
        # RL.q_table.to_excel(writer)
        # writer.save()
        
    print('game over')
    print(RL.q_table)


if __name__ == "__main__":
    env = Game()
    RL = QLearning(
        actions = list(range(env.n_actions)), 
        learning_rate = ALPHA,
        reward_decay = GAMMA,
        e_greedy = EPSILON
    )
    update(env, RL)