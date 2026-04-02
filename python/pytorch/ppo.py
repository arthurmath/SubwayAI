import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import asyncio
from utils import device, format


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
        
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy



class Agent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        # Ensure that simultaneous updates from multiple agents don't clash
        self.update_lock = asyncio.Lock()
        
    
    def act_play(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            action_probs = self.policy_old.actor(state_t)
            action = torch.argmax(action_probs)

        return action.item()


    def act_train(self, state, buffer):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state_t)

        buffer.states.append(state_t)
        buffer.actions.append(action)
        buffer.logprobs.append(action_logprob)
        buffer.state_values.append(state_val)

        return action.item()
        

    async def train(self, buffer):
        async with self.update_lock:
            if len(buffer.states) == 0:
                return

            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            # Normalize rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            if len(rewards) > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            
            old_states, old_actions, old_logprobs, old_state_values = format(buffer)
            
            advantages = rewards.detach() - old_state_values.detach()
            
            # Optimize policy for K epochs
            for _ in range(self.epochs):
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                state_values = torch.squeeze(state_values)
                if len(state_values.shape) == 0:
                    state_values = state_values.unsqueeze(0)
                
                ratios = torch.exp(logprobs - old_logprobs.detach())
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())
            buffer.clear()

