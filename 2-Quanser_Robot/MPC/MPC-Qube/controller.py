import numpy as np
import copy
from Hive import Hive
from Hive import Utilities

from optimizers import RandomOptimizer, CEMOptimizer

class MPC(object):

    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, env, config):
        self.env = env
        mpc_config = config["mpc_config"]
        self.horizon = mpc_config["horizon"]
        self.numb_bees = mpc_config["numb_bees"]
        self.max_itrs = mpc_config["max_itrs"]
        self.gamma = mpc_config["gamma"]
        self.action_low = mpc_config["action_low"]
        self.action_high = mpc_config["action_high"]

        self.action_dim = 1
        self.particle = 1
        self.popsize = 1000

        self.evaluator = Evaluator(self.gamma)

        self.type = 'CEM'

        self.optimizer = MPC.optimizers["Random"](sol_dim=self.horizon * self.action_dim, popsize = self.popsize,upper_bound=np.array([12]),lower_bound=np.array([-12]),
                                                max_iters=self.max_itrs,num_elites=20,epsilon=0.001,alpha=0.1)
        self.optimizer.setup(self.qube_cost_function)
        self.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, state, dynamic_model):
        '''
        Optimize the action by Artificial Bee Colony algorithm
        :param state: (numpy array) current state
        :param dynamic_model: system dynamic model
        :return: (float) optimal action
        '''
        self.model = dynamic_model
        self.state = state
        #MPC.cnt += 1
        #print(MPC.cnt)
        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        if self.type == "CEM":
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        else:
            pass
        action = soln[0]
        return action

    def qube_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """

        # TODO: may be able to change to tensor like pets
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)
        
        for t in range(self.horizon):
            #print(t)
            action = actions[:, t, :]  # numpy array (batch_size x action dim)
            #print(state)
            #print(action)
            #print(state.shape)
            #print(action.shape)
            input_data = np.concatenate((state,action),axis=1)

            """state_next = []
            for i in range(len(state)):
                state_next.append(self.model.predict(input_data[i]))
            state_next = np.array(state_next)
            state_next = self.model.predict(input_data) + state"""

            state_next = []
            for i in range(len(state)):
                #state_next.append(np.zeros(5))
                #print(state[i].shape)
                #print(action[i].shape)
                input_data = np.concatenate((state[i],action[i]),axis=0)
                #print(input_data.shape)
                next_ = self.model.predict(input_data)[0] + state[i]
                state_next.append(next_)
            state_next = np.array(state_next)

            cost = self.qube_cost(state_next, action)  # compute cost
            costs -= cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def qube_cost(self, state, action_n, env_cost=False, obs=True):
        """ d
        Calculate the cartpole env cost given the state

        Parameters:
        ----------
            @param numpy array - state : size should be (batch_size x state dim)
            @param numpy array - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        #print(state.shape)
        reward = []
        for i in range(len(state)):
            reward.append(float(self.get_one_cost(state[i],action_n[i])))
        #print(reward)
        #print(reward.shape)
        #print(reward)
        reward = np.array(reward)
        #print(reward)
        return reward

    def get_one_cost(self, state, action_n):
        cos_th, sin_th, cos_al, sin_al, th_d, al_d = state
        #print(cos_th.shape)
        cos_th = min(max(cos_th, -1), 1)
        #for cos in cos_al:
        cos_al = min(max(cos_al, -1), 1)
        #print(cos_th.shape)
        al=np.arccos(cos_al)
        th=np.arccos(cos_th)
        al_mod = al % (2 * np.pi) - np.pi
        #print(al_mod.shape)
        action = action_n * 5
        cost = al_mod**2 + 5e-3*al_d**2 + 1e-1*th**2 + 2e-2*th_d**2 + 3e-3*action**2  
        #print("cost:",cost)
        #print(cost)
        #print(cost.shape)
        reward = cost*0.0005
        #reward = cost
        #print(reward[0])
        #print(reward.shape)
        return -reward[0]


class Evaluator(object):
    def __init__(self, gamma=0.8):
        self.gamma = gamma

    def update(self, state, dynamic_model):
        self.state = state
        self.dynamic_model = dynamic_model

    def evaluate(self, actions):
        actions = np.array(actions)
        #print("action.shape",actions.shape)
        horizon = actions.shape[0]
        rewards = 0
        state_tmp = self.state.copy()
        for j in range(horizon):
            input_data = np.concatenate( (state_tmp,[actions[j]]) )
            state_dt = self.dynamic_model.predict(input_data)
            state_tmp = state_tmp + state_dt[0]
            rewards -= (self.gamma ** j) * self.get_reward(state_tmp, actions[j])
        return rewards

    def get_reward(self,obs, action_n):
        cos_th, sin_th, cos_al, sin_al, th_d, al_d = obs
        cos_th = min(max(cos_th, -1), 1)
        cos_al = min(max(cos_al, -1), 1)
        al=np.arccos(cos_al)
        th=np.arccos(cos_th)
        al_mod = al % (2 * np.pi) - np.pi
        action = action_n * 5
        cost = al_mod**2 + 5e-3*al_d**2 + 1e-1*th**2 + 2e-2*th_d**2 + 3e-3*action**2  
        reward = np.exp(-cost)*0.02
        return reward

