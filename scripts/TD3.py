import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 4*state_dim)
        self.layer_2 = nn.Linear(4*state_dim, 2*state_dim)
        self.layer_3 = nn.Linear(2*state_dim, state_dim)
        self.layer_4 = nn.Linear(state_dim, state_dim//2)
        self.layer_5 = nn.Linear(state_dim//2, 2*hidden_dim)
        self.layer_6 = nn.Linear(2*hidden_dim, hidden_dim)
        self.layer_7 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_8 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        s = F.relu(self.layer_3(s))
        s = F.relu(self.layer_4(s))
        s = F.relu(self.layer_5(s))
        s = F.relu(self.layer_6(s))
        s = F.relu(self.layer_7(s))
        a = self.tanh(self.layer_8(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 4*state_dim)
        self.layer_2 = nn.Linear(4*state_dim, 2*state_dim)
        self.layer_3 = nn.Linear(2*state_dim, state_dim)
        self.layer_4 = nn.Linear(state_dim, state_dim//2)
        self.layer_5 = nn.Linear(state_dim//2, 2*hidden_dim)
        self.layer_6 = nn.Linear(2*hidden_dim, hidden_dim)
        self.layer_7_s = nn.Linear(hidden_dim, hidden_dim)
        self.layer_7_a = nn.Linear(1, hidden_dim)
        self.layer_8 = nn.Linear(hidden_dim, 1)
        
        self.layer_9 = nn.Linear(state_dim, 4*state_dim)
        self.layer_10 = nn.Linear(4*state_dim, 2*state_dim)
        self.layer_11 = nn.Linear(2*state_dim, state_dim)
        self.layer_12 = nn.Linear(state_dim, state_dim//2)
        self.layer_13 = nn.Linear(state_dim//2, 2*hidden_dim)
        self.layer_14 = nn.Linear(2*hidden_dim, hidden_dim)
        self.layer_15_s = nn.Linear(hidden_dim, hidden_dim)
        self.layer_15_a = nn.Linear(1, hidden_dim)
        self.layer_16 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        s1 = F.relu(self.layer_2(s1))
        s1 = F.relu(self.layer_3(s1))
        s1 = F.relu(self.layer_4(s1))
        s1 = F.relu(self.layer_5(s1))
        s1 = F.relu(self.layer_6(s1))
        self.layer_7_s(s1)
        self.layer_7_a(a)
        s11 = torch.mm(s1, self.layer_7_s.weight.data.t())
        s12 = torch.mm(a, self.layer_7_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_7_a.bias.data)
        q1 = self.layer_8(s1)
        
        s2 = F.relu(self.layer_9(s))
        s2 = F.relu(self.layer_10(s2))
        s2 = F.relu(self.layer_11(s2))
        s2 = F.relu(self.layer_12(s2))
        s2 = F.relu(self.layer_13(s2))
        s2 = F.relu(self.layer_14(s2))
        self.layer_15_s(s2)
        self.layer_15_a(a)
        s21 = torch.mm(s2, self.layer_15_s.weight.data.t())
        s22 = torch.mm(a, self.layer_15_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_15_a.bias.data)
        q2 = self.layer_16(s2)
        
        return q1,q2

class TD3(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.hidden_size=args.hidden_size
        self.policy_freq = args.policy_freq

        self.policy_type = args.policy

        #self.device = torch.device("cuda" if args.cuda else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #initialize the actor network
        self.actor = Actor(num_inputs, action_space, self.hidden_size).to(self.device)
        self.actor_target = Actor(num_inputs, action_space, self.hidden_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        #initialize the critic network
        self.critic = Critic(num_inputs, action_space, self.hidden_size).to(self.device)
        self.critic_target = Critic(num_inputs, action_space, self.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def select_action(self, state):#, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update_parameters(self, memory, batch_size,i):
        max_Q=float('-inf')
        av_Q=0
        av_loss=0
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device).view(-1, 1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).view(-1, 1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).view(-1, 1)

        next_action = self.actor_target(next_state_batch)

        noise = torch.randn_like(action_batch).to(self.device)
        noise = noise.clamp(-0.5,0.5)
        next_action=(next_action+noise).clamp(-0.7853981633974483,0.7853981633974483)

        target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        av_Q += torch.mean(target_Q)
        max_Q = torch.max(target_Q)
        target_Q = reward_batch + ((torch.ones_like(mask_batch) - mask_batch)* self. gamma *target_Q).detach() # here I should change 1 to same size of mask_batch

        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        if i%self.policy_freq == 0:
            actor_grad,_ = self.critic(state_batch,self.actor(state_batch))
            actor_grad = -actor_grad.mean()
            self.actor_optimizer.zero_grad()
            actor_grad.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        av_loss+=loss
        av_loss=av_loss/(i+1)
        return av_loss, av_Q, max_Q

    # Save model parameters
    def save_checkpoint(self, directory, filename, i):
        torch.save(self.actor.state_dict(), "%s/%s/%s_actor_%d.pth" % (directory, filename, filename, i))
        torch.save(self.critic.state_dict(), "%s/%s/%s_critic_%d.pth" % (directory, filename, filename, i))
    

    # Load model parameters
    def load_checkpoint(self, directory, filename):
        self.actor.load_state_dict(torch.load("%s/%s/%s_actor.pth" % (directory, filename, filename), weights_only=True))
        self.critic.load_state_dict(torch.load("%s/%s/%s_critic.pth" % (directory, filename, filename), weights_only=True))