from environment import SimpleGridWorld
from agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def train():
    """
    训练主函数
    实现了完整的训练循环，包括:
    - 环境交互
    - 经验收集
    - 模型训练
    - 性能监控
    - 模型保存
    """
    # 创建环境和智能体
    env = SimpleGridWorld(size=20)
    agent = DQNAgent()
    episodes = 2000  # 训练回合数
    batch_size = 128  # 批次大小
    scores = []  # 记录每个回合的得分
    
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    best_score = float('-inf')
    target_update_frequency = 5  # 目标网络更新频率
    
    # 使用滑动窗口来判断是否收敛
    window_size = 100
    avg_scores = []
    
    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        max_steps = 400  # 每个回合的最大步数
        
        # 单个回合的循环
        for step in range(max_steps):
            # 选择动作并与环境交互
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # 存储经验并训练
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            if done:
                break
        
        scores.append(total_reward)
        
        # 计算滑动平均分数
        if len(scores) >= window_size:
            avg_score = np.mean(scores[-window_size:])
            avg_scores.append(avg_score)
            
            # 保存最佳模型
            if avg_score > best_score:
                best_score = avg_score
                torch.save(agent.model.state_dict(), 'models/best_model.pth')
        
        # 定期更新目标网络
        if episode % target_update_frequency == 0:
            agent.update_target_model()
        
        # 输出训练进度
        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            torch.save(agent.model.state_dict(), 'models/latest_model.pth')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, label='Raw Scores')
    if avg_scores:
        plt.plot(range(window_size-1, len(scores)), avg_scores, label='Average Scores')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == "__main__":
    train() 