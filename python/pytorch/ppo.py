import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import asyncio
from utils import device, format


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(ActorCritic, self).__init__()
        
        actor_layers = []
        last_dim = state_dim
        for layer_dim in layers:
            actor_layers.append(nn.Linear(last_dim, layer_dim))
            actor_layers.append(nn.Tanh())
            last_dim = layer_dim
        actor_layers.append(nn.Linear(last_dim, action_dim))
        actor_layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        last_dim = state_dim
        for layer_dim in layers:
            critic_layers.append(nn.Linear(last_dim, layer_dim))
            critic_layers.append(nn.Tanh())
            last_dim = layer_dim
        critic_layers.append(nn.Linear(last_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
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
    def __init__(self, state_dim, action_dim, layers, lr_actor, lr_critic, gamma, epochs, eps, c1, c2, gae_lambda, batch_size):
        self.epochs = epochs
        self.gamma = gamma
        self.eps = eps
        self.c1 = c1
        self.c2 = c2
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        self.policy = ActorCritic(state_dim, action_dim, layers).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, layers).to(device)
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
            state_val = self.policy_old.critic(state_t)
            action = torch.argmax(action_probs)
            # action = Categorical(action_probs).sample()

        return action.item(), action_probs.tolist(), state_val.item()


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

            old_states, old_actions, old_logprobs, old_state_values = format(buffer)

            # GAE (Generalized Advantage Estimation)
            values_list = old_state_values.detach().cpu().tolist()
            advantages_list = [0.0] * len(buffer.rewards)
            gae = 0.0
            for i in reversed(range(len(buffer.rewards))):
                is_terminal = buffer.is_terminals[i]
                next_value = 0.0 if (is_terminal or i == len(buffer.rewards) - 1) else values_list[i + 1]
                delta = buffer.rewards[i] + self.gamma * next_value - values_list[i]
                gae = delta + self.gamma * self.gae_lambda * (0.0 if is_terminal else gae)
                advantages_list[i] = gae

            advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)
            returns = advantages + old_state_values.detach()

            # Normalize advantages 
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Optimize policy for K epochs with minibatches
            n_samples = len(old_states)
            batch_size = min(self.batch_size, n_samples)

            for _ in range(self.epochs):
                indices = torch.randperm(n_samples, device=device)
                for start in range(0, n_samples, batch_size):
                    idx = indices[start:start + batch_size]
                    mb_states = old_states[idx]
                    mb_actions = old_actions[idx]
                    mb_logprobs = old_logprobs[idx]
                    mb_advantages = advantages[idx]
                    mb_returns = returns[idx]

                    logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                    state_values = torch.squeeze(state_values)
                    if len(state_values.shape) == 0:
                        state_values = state_values.unsqueeze(0)

                    ratios = torch.exp(logprobs - mb_logprobs.detach())

                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * mb_advantages

                    loss = -torch.min(surr1, surr2) + self.c1 * self.loss(state_values, mb_returns) - self.c2 * dist_entropy

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())
            buffer.clear()
