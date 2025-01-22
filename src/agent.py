import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNAgent:
    """
    DQN（Deep Q-Network）智能体类
    
    设计思路:
    1. 网络架构
       - 采用双网络架构(主网络和目标网络)来提高训练稳定性,主要有以下几个原因:
         1. 避免目标值不稳定:
            - 如果只用一个网络,目标Q值(在网格世界中表示智能体在当前位置选择某个动作后,最终到达目标位置能获得的总奖励。比如在(3,4)位置向上走一步,最终能获得多少奖励)会随着网络更新而变化,导致训练不稳定
            - 使用目标网络可以提供稳定的目标值,就像考试需要标准答案一样
         2. 减少过度估计:
            - 单个网络容易高估Q值,导致策略不够谨慎
            - 目标网络的滞后更新可以抑制Q值高估问题
         3. 打破相关性:
            - 主网络频繁更新会导致连续样本间的相关性增强
            - 目标网络保持固定一段时间,可以降低样本相关性
         4. 提升收敛性:
            - 稳定的学习目标有助于网络参数的收敛
            - 类似于教学中,稳定的考核标准有助于学生掌握知识
       - 主网络用于实时学习和动作选择,就像学生每天都在学习新知识
       - 目标网络提供稳定的学习目标,就像标准答案让学习有明确的方向
       - 网络结构为4层全连接网络,使用ReLU激活函数
    
    2. 探索与利用
       - 使用epsilon-greedy策略平衡探索和利用
       - epsilon从1.0开始,随训练进度逐渐衰减到0.02
       - 确保智能体在早期充分探索,后期更多利用学到的经验
    
    3. 经验回放机制
       - 使用deque存储最近20000条经验
       - 随机采样batch进行学习,打破经验间的相关性
       - 每条经验包含(状态,动作,奖励,下一状态,是否结束)
    
    4. 学习算法
       - 采用Q-learning算法的深度学习版本
       - 使用TD误差作为损失函数
       - 通过Adam优化器更新网络参数
       - 使用梯度裁剪防止梯度爆炸
    """
    def __init__(self, state_size=2, action_size=4):
        """
        初始化DQN智能体
        参数:
            state_size: 状态维度（这里是2，表示相对位置的x和y）
            action_size: 动作数量（这里是4，表示上下左右）
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # 经验回放缓冲区
        self.gamma = 0.99    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.02  # 最小探索率
        self.epsilon_decay = 0.997  # 探索率衰减
        self.learning_rate = 0.0003  # 学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义Q网络结构 (主网络)
        # 这个网络用于实时学习和更新,负责:
        # 1. 在选择动作时预测各个动作的Q值
        # 2. 在训练时不断更新参数以优化策略
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        
        # 定义目标网络
        # 这个网络用于提供稳定的学习目标,负责:
        # 1. 在训练时计算目标Q值,使学习更稳定
        # 2. 参数固定一段时间后才会更新(从主网络复制)
        # 3. 避免Q值估计不稳定的问题
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        
        # 初始化目标网络
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """
        将经验存储到回放缓冲区
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根据当前状态选择动作
        使用epsilon-greedy策略平衡探索与利用:
        - epsilon概率随机探索
        - 1-epsilon概率选择Q值最大的动作
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
    
    def update_target_model(self):
        """
        更新目标网络的参数
        将主网络的参数复制到目标网络,实现软更新
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
    def replay(self, batch_size=32):
        """
        从经验回放缓冲区中随机采样并学习
        实现了DQN的核心学习算法:
        1. 随机采样经验batch
        2. 计算目标Q值
        3. 计算当前Q值
        4. 反向传播更新网络
        5. 衰减探索率
        """
        if len(self.memory) < batch_size:
            return
        
        # 随机采样经验
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
        
        # 计算当前Q值和损失
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = self.criterion(current_q.squeeze(), target_q)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 