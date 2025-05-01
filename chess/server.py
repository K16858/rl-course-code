import torch
import random
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
from MicroChess import MicroChess 
from dqn_train import DQNModel

# DQNエージェント
class DQNAgent:
    def __init__(self, state_shape, hidden_channels ,action_size, num_res_blocks=20, model_path="dqn_microchess_20250218_185425.pth"):
        self.state_shape = state_shape  # (channels, height, width)
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks
        self.action_size = action_size
        
        # モデルの読み込み
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNModel(state_shape[0], action_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def select_action(self, state, valid_moves):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # 有効な手のindexを作成
        valid_moves_indices = []
        for move in valid_moves:
            # 移動元と移動先の座標から一意のインデックスを計算
            from_x, from_y, to_x, to_y = move
            index = from_x * 4 * 5 * 4 + from_y * 5 * 4 + to_x * 4 + to_y
            valid_moves_indices.append((move, index))

        # 有効な手のQ値をフィルタリング
        valid_q_values = {}
        for move, index in valid_moves_indices:
            if 0 <= index < len(q_values):  # インデックスの範囲チェック
                valid_q_values[tuple(move)] = q_values[index]

        if not valid_q_values:
            return random.choice(valid_moves)  # バックアップのランダム選択

        # Q値が最大の手を選択
        best_move = max(valid_q_values.items(), key=lambda x: x[1])[0]
        return list(best_move)  # タプルをリストに変換して返す

# FastAPIアプリの定義
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルの読み込み
MODEL_PATH = "dqn_microchess_20250412_143803.pth"
action_size = 400  # すべての可能な行動数
hidden_channels = 128
num_res_blocks = 10
state_shape = (1, 5, 4)  # PyTorchでは (channels, height, width)

agent = DQNAgent(state_shape, hidden_channels, action_size, num_res_blocks=num_res_blocks, model_path=MODEL_PATH)

# リクエストモデルの定義
class MoveRequest(BaseModel):
    state_3d: List[List[List[float]]]  # 3Dリスト (1, 5, 4)
    valid_moves: List[Tuple[int, int, int, int]]  # [(x, y, x1, y1), ...]

@app.post("/move/")
def get_best_move(request: MoveRequest):
    state = np.array(request.state_3d)
    
    # MicroChessインスタンスを作成し、現在の盤面をセット
    chess = MicroChess()
    chess.setBoard(state[0])  # state[0]で2次元配列を取得
    chess.showBoard()
    
    chess.player = -1

    # MicroChessクラスから有効な手のリストを取得
    valid_moves = chess.getValidMovesList()
    print(valid_moves)
    
    if not valid_moves:
        return {"error": "No valid moves available"}

    best_move = agent.select_action(state, valid_moves)
    return {"best_move": best_move}

# サーバーの起動
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
