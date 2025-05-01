import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from MicroChess import MicroChess
import matplotlib.pyplot as plt
from datetime import datetime


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの構築
class DQNModel(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# リプレイバッファ
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# DQNエージェント
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # (channels, height, width)
        self.action_size = action_size
        self.memory = ReplayBuffer(max_size=10000)
        self.gamma = 0.95  # 割引率
        self.epsilon = 1.0 # 探索率
        self.epsilon_min = 0.1 # 最小探索率
        self.epsilon_decay = 0.999 # 探索率の減衰率
        self.batch_size = 64 # バッチサイズ
        self.learning_rate = 0.001 # 学習率
        
        # メインネットワークとターゲットネットワーク
        self.model = DQNModel(state_shape[0], action_size).to(device)
        self.target_model = DQNModel(state_shape[0], action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # ターゲットネットワークは学習しない

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.loss_history = []
        self.update_target_freq = 10  # 10エピソードごとに更新
    
    def select_action(self, state, valid_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_moves)  # ランダム行動
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # バッチ次元を追加
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # 有効な行動のQ値をフィルタリング
        valid_q_values = {move: q_values[idx] for idx, move in enumerate(valid_moves)}

        if not valid_q_values:
            return random.choice(valid_moves)  # バックアップのランダム選択

        return max(valid_q_values, key=valid_q_values.get)
    
    def train(self):
        if self.memory.size() < self.batch_size:
            return
        
        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # 現在のQ値
        current_q = self.model(states).gather(1, actions)
        
        # 次の状態のQ値 (ターゲットネットワークを使用)
        with torch.no_grad():
            # メインネットワークで行動を選択
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # ターゲットネットワークでQ値を評価
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # 損失計算
        loss = self.criterion(current_q, target_q)
        self.loss_history.append(loss.item())
        
        # バックプロパゲーション
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # εの減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

def plot_loss_history(loss_history, save_path='loss_history.png', window_size=100):
    """学習中のロスの履歴をプロットして保存する"""
    plt.rcParams['font.family'] = 'MS Gothic'

    plt.figure(figsize=(12, 7))
    
    # 生データのプロット
    plt.plot(loss_history, alpha=0.3, color='blue', label='Raw Loss')
    
    # 移動平均の計算とプロット
    if len(loss_history) >= window_size:
        moving_avg = [
            sum(loss_history[i:i+window_size]) / window_size 
            for i in range(len(loss_history) - window_size + 1)
        ]
        plt.plot(range(window_size-1, len(loss_history)), 
                moving_avg, 
                color='red', 
                label=f'Moving Average (window={window_size})')
    
    plt.xlabel('ステップ', fontsize=12)
    plt.ylabel('損失', fontsize=12)
    
    # スケールの調整（必要に応じてlog scaleに）
    if max(loss_history) / min(loss_history) > 100:
        plt.yscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # グラフの余白調整
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# DQNの学習
def train_dqn(episodes=1000):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"dqn_microchess_{timestamp}.pth"
    loss_plot_filename = f"loss_history_{timestamp}.png"

    env = MicroChess()
    action_size = 400  # すべての可能な行動数
    state_shape = (1, 5, 4)  # PyTorchでは (channels, height, width)
    agent = DQNAgent(state_shape, action_size)
    
    print("学習開始")
    for episode in range(episodes):
        env.reset()
        state = env.getBoard().reshape(state_shape)
        total_reward = 0
        
        while not env.done:
            valid_moves = env.getValidMovesList()
            if not valid_moves:
                break
            
            # 行動の選択
            action = agent.select_action(state, valid_moves)
            x, y, x1, y1 = action
            
            # 行動の実行
            prev_piece = env.getBoard()[x1, y1]
            env.move(x, y, x1, y1)
            next_state = env.getBoard().reshape(state_shape)

            # 報酬関数
            reward = -0.5  # 毎ターンのペナルティ
            if prev_piece != 0:
                reward += 0.5  # 駒を取ったらボーナス
            if env.done:
                reward += 5 if env.winner == 1 else -5

            # アクションをインデックスに変換
            action_index = valid_moves.index(action)
            
            # リプレイバッファに保存
            agent.memory.add((state, action_index, reward, next_state, env.done))
            
            state = next_state
            total_reward += reward
            
            # 学習
            agent.train()
        
        if episode % agent.update_target_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}",  f"Loss: {agent.loss_history[-1] if agent.loss_history else 'N/A'}")
    
    # 学習済みモデルの保存
    torch.save(agent.model.state_dict(), model_filename)
    print(f"モデルを {model_filename} として保存")
    
    if agent.loss_history:
        plot_loss_history(agent.loss_history, save_path=loss_plot_filename)
        print(f"損失のグラフを {loss_plot_filename} として保存")

# --- 実行 ---
if __name__ == "__main__":
    train_dqn(episodes=500)
