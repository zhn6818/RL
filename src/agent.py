import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size=2, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # 增加记忆容量
        self.gamma = 0.99    # 保持不变
        self.epsilon = 1.0   # 初始探索率
        self.epsilon_min = 0.02  # 略微提高最小探索率
        self.epsilon_decay = 0.997  # 减缓探索率衰减
        self.learning_rate = 0.0003  # 降低学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用更大的网络
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),  # 增加网络容量
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)
        
        # 使用目标网络计算下一个状态的Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 