import numpy as np
import copy
from Hive import Hive
from Hive import Utilities

from optimizers import RandomOptimizer, CEMOptimizer


class MPC(object):

    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}
    cnt = 0

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

        self.type = "CEM"

        self.optimizer = MPC.optimizers["Random"](sol_dim=self.horizon, popsize = self.popsize,upper_bound=np.array([12]),lower_bound=np.array([-12]),
                                                max_iters=self.max_itrs,num_elites=20,epsilon=0.001,alpha=0.1)
        self.optimizer.setup(self.cartpole_cost_function)
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
        #self.evaluator.update(state, dynamic_model)

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
        
        #print("Solution: ",optimizer.solution[0])
        #print("Fitness Value ABC: {0}".format(optimizer.best))
        # Uncomment this if you want to see the performance of the optimizer
        #Utilities.ConvergencePlot(cost)

    def cartpole_cost_function(self, actions):
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
        #print(state.shape)
        #print(actions.shape)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)
            #print(state)
            #print(action)
            #print(state.shape)
            #print(action.shape)
            
            #print(input_data.shape)
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
            #print(state_next.shape)
            #state_next = self.model.predict(input_data) + state

            cost = self.cartpole_cost_2(state_next, action)  # compute cost
            #print(cost.shape)
            #print(cost.type)
            #print(self.gamma)
            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def cartpole_cost(self, state, action, env_cost=False, obs=True):
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
        if not obs:
            x = state[:, 0]
            x_dot = state[:, 1]
            theta = state[:, 2]
            theta_dot = state[:, 3]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
        else:
            # self.add_bound = 0.8
            x = state[:, 0]
            x_dot = state[:, 3]
            cos_theta = state[:, 2]
            # todo: initially the GP may predict -1.xxx for cos
            # cos_theta[cos_theta < -1] = -1
            # cos_theta[cos_theta > 1] = 1
            sin_theta = state[:, 1]
            theta_dot = state[:, 4]
        #print(x.shape)
        #print(x)
        action = action.squeeze()

        length = 0.5 # pole length
        x_tip_error = x - length*sin_theta
        y_tip_error = length - length*cos_theta
        reward = np.exp(-(x_tip_error**2 + y_tip_error**2)/length**2)

        self.action_cost = True
        self.x_dot_cost = True

        if self.action_cost:
            reward += -0.01 * action**2

        if self.x_dot_cost:
            reward += -0.001 * x_dot**2

        cost = -reward

        return cost

    def cartpole_cost_2(self, state, action_n, env_cost=False, obs=True):
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
        return reward

    def get_one_cost(self, state, action_n):
        x, sin_th, cos_th, x_dot, theta_dot = state
        cos_th = min(max(cos_th, -1), 1)
        reward = -cos_th + 1
        #print(reward)
        return -reward


class Evaluator(object):
    def __init__(self, gamma=0.8):
        self.gamma = gamma

    def update(self, state, dynamic_model):
        self.state = state
        self.dynamic_model = dynamic_model

    def evaluate(self, actions):
        actions = np.array(actions)
        horizon = actions.shape[0]
        rewards = 0
        state_tmp = self.state.copy()
        for j in range(horizon):
            input_data = np.concatenate( (state_tmp,[actions[j]]) )
            state_dt = self.dynamic_model.predict(input_data)
            state_tmp = state_tmp + state_dt[0]
            rewards -= (self.gamma ** j) * self.get_reward(state_tmp, actions[j])
        return rewards

    # need to change this function according to different environment
    def get_reward(self,obs, action_n):
        x, sin_th, cos_th, x_dot, theta_dot = obs
        cos_th = min(max(cos_th, -1), 1)
        reward = -cos_th + 1
        return reward

