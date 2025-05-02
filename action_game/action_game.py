import pygame
import sys
import time

class Player:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.velocity = 0
        self.speed = 5
        self.jump = -15
        self.on_ground = False
    
    def update(self, gravity):
        self.velocity += gravity
        self.rect.y += self.velocity
        self.on_ground = False
    
    def move_left(self):
        self.rect.x -= self.speed
    
    def move_right(self):
        self.rect.x += self.speed
    
    def jump_action(self):
        if self.on_ground:
            self.velocity = self.jump
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Obstacle:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
    
    def draw(self, screen, camera_x):
        pygame.draw.rect(screen, self.color, (self.rect.x - camera_x, self.rect.y, self.rect.width, self.rect.height))

class Hole:
    def __init__(self, x, y, width):
        self.rect = pygame.Rect(x, y, width, 50)
    
    def is_player_over(self, player_rect, camera_x):
        hole_rect = pygame.Rect(self.rect.x - camera_x, self.rect.y, self.rect.width, self.rect.height)
        return player_rect.centerx > hole_rect.left and player_rect.centerx < hole_rect.right
    
    def draw(self, screen, camera_x, background_color):
        pygame.draw.rect(screen, background_color, (self.rect.x - camera_x, self.rect.y, self.rect.width, self.rect.height))

class Goal:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
    
    def draw(self, screen, camera_x):
        pygame.draw.rect(screen, self.color, (self.rect.x - camera_x, self.rect.y, self.rect.width, self.rect.height))

class Game:
    def __init__(self, training_mode=False, visual_mode=False):
        pygame.init()
        self.training_mode = training_mode
        self.visual_mode = visual_mode
        
        if not training_mode or visual_mode:
            self.WIDTH, self.HEIGHT = 800, 600
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("シンプル横スクロールゲーム")
        else:
            self.WIDTH, self.HEIGHT = 800, 600
            self.screen = None
        
        self.clock = pygame.time.Clock()
        
        # 色の定義
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)
        self.YELLOW = (255, 255, 0)
        
        # 重力
        self.GRAVITY = 0.8
        
        # ゲーム要素の初期化
        self.init_game()
        
        # カメラ位置（横スクロール用）
        self.camera_x = 0
        
        # ゲーム状態
        self.running = True
        self.goal_reached = False
        self.game_over = False
            
        # 時間関係の変数
        self.start_time = time.time()  # 開始時間を記録
        self.elapsed_time = 0  # 経過時間
    
    def init_game(self):
        # プレイヤーの設定
        self.player = Player(100, self.HEIGHT - 100, 30, 50, self.BLUE)
        
        # 床（地面）
        self.ground = pygame.Rect(0, self.HEIGHT - 50, 3000, 50)
        
        # 障害物リスト
        self.obstacles = [
            Obstacle(300, self.HEIGHT - 100, 100, 50, self.RED),
            Obstacle(500, self.HEIGHT - 100, 100, 20, self.RED),
            Obstacle(700, self.HEIGHT - 150, 150, 20, self.RED),
            Obstacle(1000, self.HEIGHT - 100, 200, 50, self.RED)
        ]
        
        # 穴リスト
        self.holes = [
            Hole(400, self.HEIGHT - 50, 80),
            Hole(800, self.HEIGHT - 50, 120),
            Hole(1200, self.HEIGHT - 50, 100)
        ]
        
        # ゴール
        self.goal = Goal(1500, self.HEIGHT - 150, 50, 100, self.YELLOW)
        
    def reset(self):
        # ゲーム状態をリセット
        self.init_game()
        self.camera_x = 0
        self.running = True
        self.goal_reached = False
        self.game_over = False
        self.start_time = time.time()
        self.elapsed_time = 0
        
        # 状態を返す（プレイヤーの位置と速度、近くの障害物の相対位置など）
        return self._get_state()

    def _get_state(self):
        # 状態を取得
        # 状態数：8
        state = [
            self.player.rect.x / self.WIDTH,  # プレイヤーのX座標（正規化）
            self.player.rect.y / self.HEIGHT,  # プレイヤーのY座標（正規化）
            self.player.velocity / 20.0,  # プレイヤーの縦方向速度（正規化）
        ]
        
        # 前方の障害物と穴の情報も追加
        nearest_obstacle_x = float('inf')
        nearest_obstacle_y = 0
        nearest_hole_x = float('inf')
        
        # 最も近い障害物を探す
        for obstacle in self.obstacles:
            rel_x = obstacle.rect.x - self.camera_x - self.player.rect.x
            if 0 < rel_x < nearest_obstacle_x:
                nearest_obstacle_x = rel_x
                nearest_obstacle_y = obstacle.rect.y
        
        # 最も近い穴を探す
        for hole in self.holes:
            rel_x = hole.rect.x - self.camera_x - self.player.rect.x
            if 0 < rel_x < nearest_hole_x:
                nearest_hole_x = rel_x
        
        # 正規化して状態に追加
        state.append(nearest_obstacle_x / self.WIDTH if nearest_obstacle_x != float('inf') else 1.0)
        state.append(nearest_obstacle_y / self.HEIGHT)
        state.append(nearest_hole_x / self.WIDTH if nearest_hole_x != float('inf') else 1.0)
        
        # ゴールまでの距離も追加
        goal_distance = (self.goal.rect.x - self.camera_x - self.player.rect.x) / self.WIDTH
        state.append(max(0, min(1.0, goal_distance)))  # 0～1の間に正規化
        
        # 地面に接地しているかどうか
        state.append(float(self.player.on_ground))
        
        return state
    
    def _calculate_reward(self):
        reward = 0
        # 進んだ距離に比例した報酬
        reward += self.camera_x * 0.01
        
        # ゴール達成でボーナス
        if self.goal_reached:
            reward += 100.0
        
        # 穴に落ちたり死んだらペナルティ
        if self.game_over:
            reward -= 50.0
            
        return reward
    
    def step(self, action):
        # 行動を実行
        if action == 0:  # 左移動
            self.player.move_left()
        elif action == 1:  # 右移動
            self.player.move_right()
            # カメラ移動の処理
            if self.player.rect.x > self.WIDTH / 2:
                self.camera_x += self.player.speed
                self.player.rect.x = self.WIDTH / 2
        elif action == 2:  # ジャンプ
            self.player.jump_action()
        
        # 物理シミュレーション
        self.player.update(self.GRAVITY)
        self.check_collisions()
        
        self.check_game_over()
        
        # 次の状態
        next_state = self._get_state()
        
        # 報酬計算
        reward = self._calculate_reward()
        
        # ゲーム終了フラグ
        done = self.goal_reached or self.game_over or not self.running
        
        return next_state, reward, done, {}
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player.jump_action()
    
    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.move_left()
        if keys[pygame.K_RIGHT]:
            self.player.move_right()
            # 画面の半分より右に行ったらカメラを動かす
            if self.player.rect.x > self.WIDTH / 2:
                self.camera_x += self.player.speed
                self.player.rect.x = self.WIDTH / 2
    
    def check_collisions(self):
        # プレイヤーが穴の上にいるか確認
        on_hole = False
        for hole in self.holes:
            if hole.is_player_over(self.player.rect, self.camera_x):
                on_hole = True
                break
        
        # 穴の上にいない場合だけ地面との衝突判定をする
        if not on_hole:
            if self.player.rect.colliderect(self.ground):
                self.player.rect.bottom = self.ground.top
                self.player.velocity = 0
                self.player.on_ground = True
        
        # 障害物との衝突判定
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle.rect.x - self.camera_x, obstacle.rect.y, obstacle.rect.width, obstacle.rect.height)
            if self.player.rect.colliderect(obstacle_rect):
                # 上からの衝突
                if self.player.velocity > 0 and self.player.rect.bottom <= obstacle_rect.top + 10:
                    self.player.rect.bottom = obstacle_rect.top
                    self.player.velocity = 0
                    self.player.on_ground = True
                # 横からの衝突
                elif self.player.rect.right > obstacle_rect.left and self.player.rect.left < obstacle_rect.left:
                    self.player.rect.right = obstacle_rect.left
                elif self.player.rect.left < obstacle_rect.right and self.player.rect.right > obstacle_rect.right:
                    self.player.rect.left = obstacle_rect.right
    
    def check_game_over(self):
        # 画面外に落ちたらゲームオーバー
        if self.player.rect.top > self.HEIGHT:
            self.game_over = True
        
            # トレーニングモードじゃないときだけ表示する
            if not self.training_mode and self.screen:
                font = pygame.font.SysFont(None, 72)
                text = font.render("GAME OVER", True, self.RED)
                self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2))
                pygame.display.flip()
                pygame.time.wait(1000)
        
        # ゴールに到達したらゲームクリア
        goal_rect = pygame.Rect(self.goal.rect.x - self.camera_x, self.goal.rect.y, self.goal.rect.width, self.goal.rect.height)
        if self.player.rect.colliderect(goal_rect):
            self.goal_reached = True
        
            # トレーニングモードじゃないときだけ表示する
            if not self.training_mode and self.screen:
                final_time = int(time.time() - self.start_time)
                font = pygame.font.SysFont(None, 72)
                text = font.render(f"GOAL! {final_time}s", True, self.BLACK)
                self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2))
                pygame.display.flip()
                pygame.time.wait(1000)
    
    def draw(self):
        # 背景を描画
        self.screen.fill(self.WHITE)
        
        # 地面を描画
        pygame.draw.rect(self.screen, self.GREEN, (self.ground.x - self.camera_x, self.ground.y, self.ground.width, self.ground.height))
        
        # 穴を描画
        for hole in self.holes:
            hole.draw(self.screen, self.camera_x, self.WHITE)
        
        # 障害物を描画
        for obstacle in self.obstacles:
            obstacle.draw(self.screen, self.camera_x)
        
        # ゴールを描画
        self.goal.draw(self.screen, self.camera_x)
        
        # プレイヤーを描画
        self.player.draw(self.screen)
        
        self.elapsed_time = int(time.time() - self.start_time)  # 秒単位で計算
        font = pygame.font.SysFont(None, 36)
        time_text = font.render(f"Time: {self.elapsed_time}s", True, self.BLACK)
        self.screen.blit(time_text, (10, 10))  # 左上に表示
    
    def run(self):
        while self.running:
            self.handle_events()
            self.handle_input()
            self.player.update(self.GRAVITY)
            self.check_collisions()
            if not self.training_mode and not self.visual_mode:
                self.draw()
            self.check_game_over()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()