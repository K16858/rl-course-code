import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from action_game import Game
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from model import DQNModel
import pygame
import os.path


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # (channels, height, width)
        self.action_size = action_size
        self.gamma = 0.99  # 割引率
        self.epsilon = 1.0 # 探索率
        self.epsilon_min = 0.1 # 最小探索率
        self.epsilon_decay = 0.995 # 探索率の減衰率
        self.batch_size = 64 # バッチサイズ
        self.learning_rate = 0.001 # 学習率
        
        self.memory = ReplayMemory(50000)
        
        # メインネットワークとターゲットネットワーク
        self.model = DQNModel(state_shape, action_size).to(device)
        self.target_model = DQNModel(state_shape, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # ターゲットネットワークは学習しない

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.loss_history = []
        self.update_target_freq = 5  # 10エピソードごとに更新
        
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
    
    plt.xlabel('steps', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    
    # スケールの調整
    if max(loss_history) / min(loss_history) > 100:
        plt.yscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # グラフの余白調整
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def train_dqn(episodes=1000, load_model=None, epsilon=None):
    # タイムスタンプは新しく生成するか、読み込むモデル名から取得
    if load_model:
        timestamp = os.path.basename(load_model).replace("dqn_action_game_", "").replace(".pth", "")
        model_filename = load_model
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"dqn_action_game_{timestamp}.pth"
    
    loss_plot_filename = f"loss_history_{timestamp}.png"

    env = Game(training_mode=True, visual_mode=True)
    action_size = 3  # すべての可能な行動数（右，左，ジャンプ）
    state_shape = 8  # 状態の形状
    agent = DQNAgent(state_shape, action_size)
    
    # モデルを読み込む処理
    if load_model and os.path.exists(load_model):
        agent.model.load_state_dict(torch.load(load_model))
        agent.target_model.load_state_dict(agent.model.state_dict())
        print(f"モデルを {load_model} から読み込み")
        
        # イプシロンを指定されていれば上書き
        if epsilon is not None:
            agent.epsilon = epsilon
            print(f"イプシロン値を {epsilon} に設定")
    else:
        print("新しくモデルを作るね～")
    
    print("学習開始")
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # 学習を終了
                    
            action = agent.select_action(state, [0,1,2])
            next_state, reward, done, info = env.step(action)
            
            # 画面の更新を確実にする
            if env.visual_mode:
                if episode % 5 == 0:  # 5エピソードに1回だけ表示を遅くする
                    pygame.time.delay(5)  # 確認用の遅延
                else:
                    pygame.time.delay(1)  # 通常は最小限の遅延
                env.draw()  # 描画関数を呼び出し
                pygame.display.flip()  # 画面を更新する
            
            agent.memory.add(state, action, reward, next_state, done)
            agent.train()    
            state = next_state
            total_reward += reward
            
        if episode % agent.update_target_freq == 0:
            agent.update_target_network()
        
        if episode % 10 == 0:
            print(f"エピソード {episode}/{episodes} - 報酬: {total_reward} - ε: {agent.epsilon:.2f}")
        
        if episode % 100 == 0:
            # モデルの保存
            torch.save(agent.model.state_dict(), model_filename)
            print(f"モデルを {model_filename} として保存")
    
    # 学習済みモデルの保存
    torch.save(agent.model.state_dict(), model_filename)
    print(f"モデルを {model_filename} として保存")
    
    if agent.loss_history:
        plot_loss_history(agent.loss_history, save_path=loss_plot_filename)
        print(f"損失のグラフを {loss_plot_filename} として保存")

# --- 実行 ---
if __name__ == "__main__":
    train_dqn(episodes=5000)
    