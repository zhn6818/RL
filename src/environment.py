import numpy as np

class SimpleGridWorld:
    def __init__(self, size=20):
        self.size = size
        self.reset()
        
    def reset(self):
        # 随机初始化智能体位置
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
        # 状态包含智能体到目标的相对位置
        return self.target_pos - self.agent_pos
    
    def step(self, action):
        # 保存旧的位置，用于计算是否更接近目标
        old_dist = np.linalg.norm(self.target_pos - self.agent_pos)
        
        # 动作空间：上(0)、下(1)、左(2)、右(3)
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
        
        # 计算奖励
        done = False
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 5.0
            done = True
        else:
            # 根据距离变化给予奖励或惩罚
            reward = (old_dist - new_dist) * 0.2
            # 添加较小的步数惩罚
            reward -= 0.005
            
        return self._get_state(), reward, done 