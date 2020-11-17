from env import Game
from env import Visualizer
import copy
import numpy as np
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

episode_num = 100
theta = 0.15

Value_table=np.zeros((10, 21)) # 创建并初始化价值矩阵
Policy=np.zeros((10, 21)) # 创建策略矩阵
prob = 0.1
red_prob = 1/3 * prob
black_prob = 2/3 * prob

def initial_policy(): #初始化策略矩阵
    for i in range(10):
        for j in range(21):
            Policy[i,j]=np.random.randint(0, 2)

def update_value_for_one_state(env, action, dealer, player, v_table):
    v = 0
    if action==0:
        for i in range(10): # 如果下一张牌抽到红牌i
            env.set(dealer, player)
            reward = env.hit_value(-i)
            if reward != -1:
                v += red_prob*(reward + v_table[dealer-1,player-i-1])
            else:
                v += red_prob * reward

        for j in range(10): #如果下一张牌抽到黑牌j
            env.set(dealer, player)
            reward = env.hit_value(j)
            if reward != -1:
                v += black_prob * (reward + v_table[dealer - 1, player + j - 1])
            else:
                # v += black_prob * (player-11 ) *reward
                v += black_prob *  reward

    if action == 1:
        for k in range(500): #模拟500次dealer的状态然后取reward的平均值
            env.set(dealer,player)
            _, reward = env.step(action)
            v += reward
        v = v/500
    return v

def policy_evaluation(env,Value_table):
    while (True):
        delta = 0
        newValue = copy.deepcopy(Value_table)
        for dealer in range(1,11):
            for player in range(1,22): # 对所有状态遍历
                action = Policy[dealer-1, player-1] # 当前状态对应的策略
                newValue[dealer - 1, player - 1]=update_value_for_one_state(env,action, dealer, player, Value_table )
                delta = max(delta, abs(Value_table[dealer - 1, player - 1] - newValue[dealer - 1, player - 1]))
        Value_table = copy.deepcopy(newValue)
        if (delta < theta):
            return delta, Value_table # 收敛了就退出循环

def policy_improvement(env,dealer, player, Value_table):
    hit = update_value_for_one_state(env,0,dealer, player, Value_table)
    stick = update_value_for_one_state(env,1,dealer, player, Value_table)
    # 比较两个动作谁的价值更高就选择哪个
    if (hit>stick):
        Policy[dealer - 1, player - 1] = 0
    else:
        Policy[dealer - 1, player - 1] = 1

if __name__ == "__main__":
    env = Game()
    initial_policy()
    result = []
    for epi in range(episode_num):
        # 策略评估
        pe,Value_table = policy_evaluation(env, Value_table)
        print(pe, epi)
        # 策略提升
        policy_stable = True # 如果对所有的状态策略都没有更新，那么提前结束
        for dealer in range(1, 11):
            for player in range(1, 22):
                old_action = Policy[dealer - 1, player - 1]
                policy_improvement(env, dealer, player, Value_table)
                if (old_action != Policy[dealer - 1, player - 1]): policy_stable=False
        # print(Policy)
        if(policy_stable):
            break

        # 计算获胜的概率
        win = 0
        draw = 0  # 平局
        win_rate = 0

        for i in range(200000):
            env.reset()
            while (True):
                d, p = env.get_state()
                action = Policy[d - 1, p - 1]
                next, reward = env.step(action)

                if (next == 'terminal'):
                    if (reward > 0): win += 1
                    if (reward == 0): draw += 1
                    win_rate = win/200000
                    break
        print(epi, win, draw, win_rate)
        result.append(win_rate)

        f = open('result', 'a')  # 打开test.txt   如果文件不存在，创建该文件。
        f.write(str(epi)+'\t'+ str(win)+'\t'+str(draw)+'\t'+str(win_rate)+'\n')
        f.close()


    # 画learning curve
    epi = [i+1 for i in range(episode_num)]
    plt.plot(epi, result)
    plt.xlabel("episode")  # x轴的标记
    plt.ylabel("win rate")
    plt.title("learning curve")
    plt.show()
    # 画Value_table图
    V_dic = defaultdict(int)
    for i in range(1, 22):
        for j in range(1, 11):
            state = (j, i)
            V_dic[state, 0] = Value_table[j - 1, i - 1]
            V_dic[state, 1] = -1000
    Visualizer.visualize(V_dic, 'V-value')
    # 画Policy图
    Visualizer.draw2d_array(Policy.transpose(), 'Optimal Policy [hit = 0, stick = 1]', True)


