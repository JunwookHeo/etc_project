# https://nodiscard.tistory.com/302

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
 
###################################
# Actor-Critic 통합 네트워크
###################################
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # 정책(action) 출력 계층
        self.action_head = nn.Linear(hidden_size, action_dim)
        # 가치(value) 출력 계층
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        value = self.value_head(x)
        return action_logits, value
 
###################################
# A2C 에이전트
###################################
class A2CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, update_steps=5):
        self.gamma = gamma
        self.update_steps = update_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ac_net = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
        # 샘플링 결과를 저장할 버퍼(짧은 n-step)
        self.states = []
        self.actions = []
        self.rewards = []
        self.done_flags = []
        
        self.action_dim = action_dim
 
    def select_action(self, state):
        # state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_t = torch.from_numpy(state)
        action_logits, _ = self.ac_net(state_t)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        self.states.append(state)
        self.actions.append(action.item())
        return action.item()
    
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.done_flags.append(done)
 
    def update(self, next_state):
        # next_state로 V(s') 계산
        if self.done_flags[-1]:
            # 종료 상태이면 V(s')=0
            next_value = 0.0
        else:
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value_t = self.ac_net(next_state_t)
                next_value = next_value_t.item()
        
        # n-step Returns 계산
        returns = []
        G = next_value
        for r, done in reversed(list(zip(self.rewards, self.done_flags))):
            if done:
                G = r  # done이면 G를 리셋
            else:
                G = r + self.gamma * G
            returns.insert(0, G)
        
        states_t = torch.FloatTensor(self.states).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        action_logits, values_t = self.ac_net(states_t)
        values_t = values_t.squeeze(1)
        
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_t)
        
        # Advantage = returns_t - values_t
        advantage = returns_t - values_t.detach()
        
        # Actor Loss = -logπ(a|s)*Advantage
        actor_loss = -(log_probs * advantage).mean()
        # Critic Loss = (V(s)-Returns)^2
        critic_loss = F.mse_loss(values_t, returns_t)
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 버퍼 초기화
        self.states = []
        self.actions = []
        self.rewards = []
        self.done_flags = []
 
def train_a2c(env_name="CartPole-v1", max_episodes=300, update_steps=5):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2CAgent(state_dim, action_dim, update_steps=update_steps)
    
    reward_history = []
    state = env.reset()[0]
    for ep in range(max_episodes):
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store_reward(reward, done)
            
            state = next_state
            total_reward += reward
            
            # 일정 스텝마다 업데이트 수행 (또는 에피소드 종료 시 수행)
            if len(agent.rewards) >= agent.update_steps or done:
                agent.update(next_state)
        
        reward_history.append(total_reward)
        if (ep+1) % 20 == 0:
            avg_reward = np.mean(reward_history[-20:])
            print(f"Episode {ep+1}, Avg Reward(last 20): {avg_reward:.2f}")
        
        # 에피소드 끝났으니 다음 에피소드 시작
        state = env.reset()[0]
 
    env.close()
 
if __name__ == "__main__":
    train_a2c()