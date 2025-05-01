import torch
import torch.nn as nn
import torch.optim as optim

# DQNモデルの構築
class DQNModel(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQNModel, self).__init__()
        self.liner = nn.Sequential(
            nn.Linear(input_channels, 512), # 入力層
            nn.ReLU(),                      # 活性化関数
            nn.Linear(512, 256),            # 隠れ層
            nn.ReLU(),                      # 活性化関数
            nn.Linear(256, action_size)     # 出力層
        )
    
    def forward(self, x):
        x = self.liner(x)
        return x
