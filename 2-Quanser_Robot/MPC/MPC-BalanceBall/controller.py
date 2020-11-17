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

        self.optimizer = MPC.optimizers["Random"](sol_dim=self.horizon*self.action_dim, popsize = self.popsize,upper_bound=np.array([0.5]),lower_bound=np.array([-0.5]),
                                                max_iters=self.max_itrs,num_elites=30,epsilon=0.001,alpha=0.1)
        self.optimizer.setup(self.balanceball_cost_function)
        self.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        #print(self.prev_sol.shape)
        #print(self.prev_sol)
        #self.prev_sol = np.concatenate((self.prev_sol,self.prev_sol),axis=1)
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])
        #self.init_var = np.concatenate((self.init_var,self.init_var),axis=1)
        #print(self.prev_sol.shape)
        #print(self.init_var.shape)

    def act(self, state, dynamic_model,mode):
        '''
        Optimize the action by Artificial Bee Colony algorithm
        :param state: (numpy array) current state
        :param dynamic_model: system dynamic model
        :return: (float) optimal action
        '''
        #self.evaluator.update(state, dynamic_model)
        self.mode = mode
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

    def balanceball_cost_function(self, actions):
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
            action = actions[:, t, :]  # numpy array (batch_size x action dim)
            #print(state)
            #print(action)
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
                if self.mode == "X":
                    input_data = np.concatenate((state[i],action[i],[0]),axis=0)
                if self.mode == "Y":
                    input_data = np.concatenate((state[i],[0],action[i]),axis=0)
                #print(input_data.shape)
                next_ = self.model.predict(input_data)[0] + state[i]
                state_next.append(next_)
            state_next = np.array(state_next)


            cost = self.balanceball_cost(state_next, action)  # compute cost
            #print(cost.shape)
            #print(cost.type)
            #print(self.gamma)
            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def balanceball_cost(self, state, action, env_cost=False, obs=True):
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
            reward.append(float(self.get_one_cost(state[i],action[i])))
        #print(reward)
        #print(reward.shape)
        #print(reward)
        reward = np.array(reward)
        return reward


    def get_one_cost(self, obs, action):
        self._state_des = np.zeros(obs.shape)
        self.Q = np.diag([1e-2, 1e-2, 1e-0, 1e-0, 1e-4, 1e-4, 1e-2, 1e-2])  # see dim of state space
        self.R = np.diag([1e-4, 1e-4])  # see dim of action space
        self.min_rew = 1e-4
        err_s = (self._state_des - obs).reshape(-1,)  # or self._state
        if self.mode == 'X':
            action = np.array([action[0],0])
        if self.mode == 'Y':
            action = np.array([0,action[0]])
        
        err_a = action.reshape(-1,)
        quadr_cost = err_s.dot(self.Q.dot(err_s)) + err_a.dot(self.R.dot(err_a))
        state_max = np.array([np.pi/4., np.pi/4., 0.15, 0.15, 4.*np.pi, 4.*np.pi, 0.5, 0.5])
        act_max = np.array([5.0, 5.0])
        obs_max = state_max.reshape(-1, )
        act_max = act_max.reshape(-1, )

        max_cost = obs_max.dot(self.Q.dot(obs_max)) + act_max.dot(self.R.dot(act_max))
        # Compute a scaling factor that sets the current state and action in relation to the worst case
        self.c_max = -1.0 * np.log(self.min_rew) / max_cost

        # Calculate the scaled exponential
        rew = np.exp(-self.c_max * quadr_cost)  # c_max > 0, quard_cost >= 0
        return -float(rew)




