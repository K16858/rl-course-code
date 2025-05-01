# MLRL-course-code
岡山大学電子計算機研究会の2025年の新歓講座で使用したコードです

## プロジェクト概要

このリポジトリには2つの強化学習プロジェクトが含まれています：

1. **シンプル横スクロールゲーム** - マリオ風の横スクロールアクションゲームをDQNで学習
2. **MicroChess** - 5×4の小さなチェスボードを使った簡易チェスゲームをDQNで学習

## 導入方法

必要なパッケージをインストール：
```bash
pip install -r requirements.txt
```

## chess
- dqn_train.py : 学習スクリプト
- evaluate : 評価スクリプト
- main.py : MicroChessのテストプレイ
- MicroChess.py : 5x4のマイクロチェス環境
- server.py : jsとの通信用ローカルサーバー
- その他 : AIとの対戦環境
モデルを学習させたのち，play.htmlを開くと作成したモデルと対戦することができます．

## action_game
- train.py : 学習スクリプト
- model.py : モデルクラスの定義ファイル
- action_game.py : 横スクロールアクションゲームの環境
