import numpy as np

class SimpleGridWorld:
    """
    简单的网格世界环境类
    实现了一个N×N的网格世界，智能体需要在其中找到目标位置
    这是一个典型的强化学习环境，包含状态空间、动作空间和奖励函数
    """
    def __init__(self, size=20):
        """
        初始化网格世界
        参数:
            size: 网格的大小，默认为20×20
        """
        self.size = size
        self.reset()
        
    def reset(self):
        """
        重置环境到初始状态
        - 随机放置智能体
        - 随机放置目标点（确保与智能体不重叠）
        返回:
            初始状态（智能体到目标的相对位置）
        """
        # 随机初始化智能体位置
        # 智能体就像是一个在棋盘上移动的棋子
        # 比如在一个20x20的棋盘上,智能体可能在坐标(3,5)的位置
        # 下面的代码就是随机选择一个位置作为智能体的起点
        # 例如可能随机生成坐标(3,5), (12,8), (0,19)等
        self.agent_pos = np.array([np.random.randint(0, self.size),
                                 np.random.randint(0, self.size)])
        # 随机初始化目标位置，确保不与智能体重叠
        while True:
            self.target_pos = np.array([np.random.randint(0, self.size), 
                                      np.random.randint(0, self.size)])
            if not np.array_equal(self.agent_pos, self.target_pos):
                break
        return self._get_state()
    
    def _get_state(self):
        """
        获取当前状态
        这个函数的主要作用是计算智能体与目标之间的相对位置向量
        通过返回目标位置减去智能体位置的向量差，可以得到:
        1. 智能体需要在x轴移动的距离和方向
        2. 智能体需要在y轴移动的距离和方向
        这种状态表示方法可以大大简化状态空间,让智能体更容易学习
        返回:
            相对位置向量 = 目标位置 - 智能体位置
            例如:如果目标在(5,3),智能体在(2,1),则返回向量(3,2)
            表示智能体需要向右移动3步,向下移动2步才能到达目标
        """
        """
        获取当前状态
        返回:
            智能体到目标的相对位置向量
        这个状态表示方法很巧妙，使用相对位置可以大大减少状态空间
        """
        return self.target_pos - self.agent_pos
    
    def step(self, action):
        """
        执行一个动作并返回结果。这是强化学习环境中的核心函数，用于:
        1. 接收智能体选择的动作(上下左右移动)
        2. 更新环境状态(移动智能体位置)
        3. 计算奖励(根据是否接近目标)
        4. 判断回合是否结束(是否到达目标)
        
        这个函数让智能体能够与环境进行交互,通过尝试不同的动作来学习最优策略。
        每次调用都会让环境前进一步,直到回合结束。
        """
        """
        执行一个动作并返回结果
        参数:
            action: 动作（0:上, 1:下, 2:左, 3:右）
        返回:
            (next_state, reward, done): 下一个状态、奖励和是否结束
        """
        # 保存旧的位置，用于计算是否更接近目标
        old_dist = np.linalg.norm(self.target_pos - self.agent_pos)
        
        # 执行动作，同时确保不会超出网格边界
        if action == 0:    # 上
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # 下
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # 左
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # 右
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
            
        # 计算新的距离
        new_dist = np.linalg.norm(self.target_pos - self.agent_pos)
        
        # 计算奖励：到达目标得高奖励，否则根据是否接近目标给予小奖励或惩罚
        done = False
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 5.0  # 到达目标的奖励
            done = True
        else:
            # 根据距离变化给予奖励或惩罚
            reward = (old_dist - new_dist) * 0.2  # 接近目标给正奖励，远离给负奖励
            reward -= 0.005  # 小惩罚以鼓励快速到达
            
        return self._get_state(), reward, done 