from environment import SimpleGridWorld
from agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def train():
    env = SimpleGridWorld(size=20)
    agent = DQNAgent()
    episodes = 2000  # 增加训练回合
    batch_size = 128  # 增加批次大小
    scores = []
    
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    best_score = float('-inf')
    target_update_frequency = 5  # 更频繁地更新目标网络
    
    # 使用滑动窗口来判断是否收敛
    window_size = 100
    avg_scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        max_steps = 400  # 增加最大步数
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            if done:
                break
        
        scores.append(total_reward)
        
        # 计算滑动平均
        if len(scores) >= window_size:
            avg_score = np.mean(scores[-window_size:])
            avg_scores.append(avg_score)
            
            # 保存最佳模型
            if avg_score > best_score:
                best_score = avg_score
                torch.save(agent.model.state_dict(), 'models/best_model.pth')
        
        # 更新目标网络
        if episode % target_update_frequency == 0:
            agent.update_target_model()
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            # 定期保存模型
            torch.save(agent.model.state_dict(), 'models/latest_model.pth')
    
    # 绘制并保存学习曲线
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