import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import random
from MicroChess import MicroChess  # MicroChess環境のインポート

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの構築 (DQN)
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

# DQNエージェント
class DQNAgent:
    def __init__(self, state_shape, action_size, model_path):
        self.state_shape = state_shape  # (channels, height, width)
        self.action_size = action_size
        
        # モデルの読み込み
        self.model = DQNModel(state_shape[0], action_size).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()  # 学習は行わない
        
    def select_action(self, state, valid_moves):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # バッチ次元を追加
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # 有効な行動のQ値をフィルタリング
        valid_q_values = {move: q_values[idx] for idx, move in enumerate(valid_moves)}

        if not valid_q_values:
            return random.choice(valid_moves)  # バックアップのランダム選択

        return max(valid_q_values, key=valid_q_values.get)

# ランダムエージェント
class RandomAgent:
    def select_action(self, state, valid_moves):
        return random.choice(valid_moves) if valid_moves else None

# DQN vs ランダムプレイヤーの対戦
def evaluate_dqn(episodes=100):
    env = MicroChess()
    action_size = 400  # すべての可能な行動数
    state_shape = (1, 5, 4)  # (channels, height, width)
    
    dqn_agent = DQNAgent(state_shape, action_size, "dqn_microchess_20250412_143803.pth")
    random_agent = RandomAgent()
    
    dqn_wins, random_wins, draws = 0, 0, 0
    
    for episode in range(episodes):
        env.reset()
        state = env.getBoard().reshape(state_shape)
        is_dqn_turn = True  # DQNが先手
        
        while not env.done:
            valid_moves = env.getValidMovesList()
            if not valid_moves:
                break

            # 行動の選択
            if is_dqn_turn:
                action = dqn_agent.select_action(state, valid_moves)
            else:
                action = random_agent.select_action(state, valid_moves)
            
            if action is None:
                break

            x, y, x1, y1 = action
            env.move(x, y, x1, y1)

            # 次の状態を取得
            state = env.getBoard().reshape(state_shape)
            is_dqn_turn = not is_dqn_turn  # ターンを交代
        
        # 勝敗のカウント
        if env.winner == 1:
            dqn_wins += 1
        elif env.winner == -1:
            random_wins += 1
        else:
            draws += 1

        print(f"Episode {episode + 1}: DQN Wins: {dqn_wins}, Random Wins: {random_wins}, Draws: {draws}")
    
    # 結果の表示
    print("\n=== 対戦結果 ===")
    print(f"DQN 勝利数: {dqn_wins} / {episodes} ({(dqn_wins / episodes) * 100:.2f}%)")
    print(f"Random 勝利数: {random_wins} / {episodes} ({(random_wins / episodes) * 100:.2f}%)")
    print(f"引き分け: {draws} / {episodes} ({(draws / episodes) * 100:.2f}%)")

# --- 実行 ---
if __name__ == "__main__":
    evaluate_dqn(episodes=10)
