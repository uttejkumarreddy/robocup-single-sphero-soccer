import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from AI.DDPG.ActorNetwork import ActorNetwork
from AI.DDPG.CriticNetwork import CriticNetwork
from AI.DDPG.ReplayBuffer import ReplayBuffer
from AI.DDPG.OUActionNoise import OUActionNoise

from Utilities.Logger import Logger

class Agent(object):
	def __init__(self, alpha, beta, input_dims, tau, gamma = 0.99,
							n_actions = 2, max_size = 1000000, 
							layer1_size = 400, layer2_size = 300, batch_size = 64):
		self.logger = Logger()
		self.logger.write("Initializing DDPG Agent with alpha {0} beta {1} input_dims {2} tau {3} gamma {4} n_actions {5} max_size {6} layer1_size {7} layer2_size {8} batch_size {9}\n".format(alpha, beta, input_dims, tau, gamma, n_actions, max_size, layer1_size, layer2_size, batch_size))
		
		self.gamma = gamma
		self.tau = tau
		self.memory = ReplayBuffer(max_size, input_dims, n_actions)
		self.batch_size = batch_size

		self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = 'Actor')
		self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = 'TargetActor')

		self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = 'Critic')
		self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = 'TargetCritic')

		self.noise = OUActionNoise(mu = np.zeros(n_actions))

		self.update_network_parameters(tau = 1)

	def choose_action(self, observation):
		self.actor.eval()
		observation = T.tensor(observation, dtype = T.float).to(self.actor.device)
		mu = self.actor(observation).to(self.actor.device)
		mu_prime = mu + T.tensor(self.noise(), dtype = T.float).to(self.actor.device)
		self.actor.train()
		return mu_prime.cpu().detach().numpy()

	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return

		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
		reward = T.tensor(reward, dtype = T.float).to(self.critic.device)
		done = T.tensor(done).to(self.critic.device)
		new_state = T.tensor(new_state, dtype = T.float).to(self.critic.device)
		action = T.tensor(action, dtype = T.float).to(self.critic.device)
		state = T.tensor(state, dtype = T.float).to(self.critic.device)

		self.target_actor.eval()
		self.target_critic.eval()
		self.critic.eval()

		target_actions = self.target_actor.forward(new_state)
		critic_value_ = self.target_critic.forward(new_state, target_actions)
		critic_value = self.critic.forward(state, action)

		target = []
		for j in range(self.batch_size):
			target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
		target = T.tensor(target).to(self.critic.device)
		target = target.view(self.batch_size, 1)

		self.critic.train()
		self.critic.optimizer.zero_grad()
		critic_loss = F.mse_loss(target, critic_value)
		critic_loss.backward()
		self.critic.optimizer.step()

		self.critic.eval()
		self.actor.optimizer.zero_grad()
		mu = self.actor.forward(state)
		self.actor.train()
		actor_loss = -self.critic.forward(state, mu)
		actor_loss = T.mean(actor_loss)
		actor_loss.backward()
		self.actor.optimizer.step()

		self.update_network_parameters()

	def update_network_parameters(self, tau = None):
		if tau is None:
			tau = self.tau

		actor_params = self.actor.named_parameters()
		critic_params = self.critic.named_parameters()
		target_actor_params = self.target_actor.named_parameters()
		target_critic_params = self.target_critic.named_parameters()

		critic_state_dict = dict(critic_params)
		actor_state_dict = dict(actor_params)
		target_critic_dict = dict(target_critic_params)
		target_actor_dict = dict(target_actor_params)

		for name in critic_state_dict:
			critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()
		self.target_critic.load_state_dict(critic_state_dict)

		for name in actor_state_dict:
			actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
		self.target_actor.load_state_dict(actor_state_dict)

	def save_models(self):
		self.actor.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_critic.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.critic.load_checkpoint()
		self.target_critic.load_checkpoint()