import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from utils import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from quanser_robots.common import GentlyTerminating

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',  
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="CartpoleSwingShort-v0", metavar='G',  # 环境名称
                     help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=0.01, metavar='G',  
                    help='max kl value (default: 0.01)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',  
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=5, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=1500, metavar='N',
                    help='batch-size')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--max_steps', type=int, default=1000, metavar='N',
                    help='maximum steps per episode')
parser.add_argument('--exp_id', type=str, default='exp-tc-04', metavar='N',
                    help='experiment id')
parser.add_argument('--log_path', type=str, default="/home/zhangxi/git/RL-teamwork/pytorch-trpo/log/", metavar='N',
                    help='the path to save log file')
args = parser.parse_args()


def PRINT_AND_LOG(message):
    print(message) 
    file_name = args.exp_id + '-seed_' + str(args.seed) + '-gamma_' + str(args.gamma) + '-maxkl_' + str(args.max_kl) + '-batchsize_' + str(args.batch_size) + ".txt"
    with open(args.log_path + args.env_name + '/' + file_name, 'a') as f:
        f.write(message + "\n")

for key in args.__dict__:
	print(f"{key}:{args.__dict__[key]}")
    message = '{}\t{}'.format(key, args.__dict__[key])
    PRINT_AND_LOG(message)

env = GentlyTerminating(gym.make(args.env_name))

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, loss_value, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    loss_policy = trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)
    return loss_value, loss_policy

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

for i_episode in range(1005):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(args.max_steps): # 限制每个episode的最大步数
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    step_average = num_steps / num_episodes + 2
    batch = memory.sample()
    loss_value, loss_policy = update_params(batch)
    ave_r = reward_batch / step_average

    if i_episode % args.log_interval == 0:
        PRINT_AND_LOG('Episode\t{}\tLast reward\t{:.4f}\tAverage_reward\t{:.4f}\tEpi_num\t{}\tAverage_step\t{}\tAverage_reward\t{}\tLoss_v\t{}\tLoss_p\t{}'.format(
            i_episode, reward_sum, reward_batch, num_episodes, step_average,ave_r,loss_value, loss_policy))

