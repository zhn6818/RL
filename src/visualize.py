import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import SimpleGridWorld
from agent import DQNAgent
import time

class GridWorldVisualizer:
    def __init__(self, env):
        self.env = env
        # 创建固定的图形窗口
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Grid World Visualization')
        
    def render(self, agent_pos, target_pos):
        # 清除当前轴的内容，而不是整个图形
        self.ax.clear()
        
        # 创建网格
        grid = np.zeros((self.env.size, self.env.size))
        
        # 标记智能体位置和目标位置
        grid[agent_pos[1], agent_pos[0]] = 1
        grid[target_pos[1], target_pos[0]] = 2
        
        # 显示网格
        self.ax.imshow(grid, cmap='coolwarm')
        self.ax.grid(True)
        self.ax.set_xticks(range(self.env.size))
        self.ax.set_yticks(range(self.env.size))
        
        # 添加图例和标题
        self.ax.set_title(f'Grid World ({self.env.size}x{self.env.size})')
        self.ax.text(-1, -1, 'Blue: Agent\nRed: Target', fontsize=10)
        
        # 在网格上标注坐标
        for i in range(self.env.size):
            for j in range(self.env.size):
                if i == agent_pos[1] and j == agent_pos[0]:
                    self.ax.text(j, i, 'A', ha='center', va='center')
                elif i == target_pos[1] and j == target_pos[0]:
                    self.ax.text(j, i, 'T', ha='center', va='center')
        
        # 更新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # 稍微减少暂停时间
    
    def close(self):
        plt.close(self.fig)

def evaluate_model(model_path, num_episodes=5, grid_size=20):
    env = SimpleGridWorld(size=grid_size)
    agent = DQNAgent()
    # 加载训练好的模型
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0  # 关闭探索，纯粹使用学习到的策略
    
    visualizer = GridWorldVisualizer(env)
    episode_stats = []  # 记录每个回合的统计信息
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            start_dist = np.linalg.norm(env.target_pos - env.agent_pos)
            
            print(f"\nEpisode {episode + 1}")
            print(f"Initial distance to target: {start_dist:.2f}")
            print(f"Agent position: {env.agent_pos}, Target position: {env.target_pos}")
            visualizer.render(env.agent_pos, env.target_pos)
            
            path = [env.agent_pos.copy()]
            
            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                path.append(env.agent_pos.copy())
                visualizer.render(env.agent_pos, env.target_pos)
                
                if done:
                    final_dist = np.linalg.norm(env.target_pos - env.agent_pos)
                    print(f"Episode finished after {steps} steps")
                    print(f"Total reward: {total_reward:.2f}")
                    print(f"Final distance to target: {final_dist:.2f}")
                    episode_stats.append({
                        'steps': steps,
                        'reward': total_reward,
                        'success': final_dist < 0.1,
                        'path_length': len(path)
                    })
                    time.sleep(0.5)
                    break
                
                if steps >= grid_size * 2:
                    print("Episode timeout")
                    episode_stats.append({
                        'steps': steps,
                        'reward': total_reward,
                        'success': False,
                        'path_length': len(path)
                    })
                    break
    finally:
        visualizer.close()  # 确保窗口被正确关闭
    
    return episode_stats

def analyze_performance(model_path, num_episodes=100, grid_size=20):
    env = SimpleGridWorld(size=grid_size)
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0
    
    success_count = 0
    step_counts = []
    rewards = []
    path_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        path_length = 0
        
        while steps < grid_size * 2:  # 根据网格大小调整最大步数
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            path_length += 1
            
            if done:
                success_count += 1
                step_counts.append(steps)
                rewards.append(total_reward)
                path_lengths.append(path_length)
                break
    
    # 显示详细的统计信息
    print(f"\nPerformance Analysis over {num_episodes} episodes:")
    print(f"Success Rate: {success_count/num_episodes*100:.2f}%")
    if step_counts:
        print(f"Average Steps to Goal: {np.mean(step_counts):.2f}")
        print(f"Min Steps: {np.min(step_counts)}")
        print(f"Max Steps: {np.max(step_counts)}")
        print(f"Average Reward: {np.mean(rewards):.2f}")
        print(f"Average Path Length: {np.mean(path_lengths):.2f}")
    
    # 绘制多个统计图表
    plt.figure(figsize=(15, 5))
    
    # 步数分布直方图
    plt.subplot(131)
    plt.hist(step_counts, bins=20)
    plt.title('Steps Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    
    # 奖励分布直方图
    plt.subplot(132)
    plt.hist(rewards, bins=20)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    
    # 路径长度分布直方图
    plt.subplot(133)
    plt.hist(path_lengths, bins=20)
    plt.title('Path Length Distribution')
    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = 'models/best_model.pth'
    
    print("Starting visual evaluation...")
    episode_stats = evaluate_model(model_path, num_episodes=5, grid_size=20)
    
    print("\nStarting performance analysis...")
    analyze_performance(model_path, num_episodes=100, grid_size=20) 