import numpy as np

class MicroChess:
    def __init__(self):
        self.board = np.zeros((5, 4), dtype=int)
        self.player = 1
        self.winner = 0
        self.done = False
        self.moves = 0
        
    def setBoard(self, board=np.array(((-1, -2, -3, -4), (-5, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 5), (4, 3, 2, 1)), dtype=int)):
        # 1:King, 2:Knight, 3:Bishop, 4:Rook, 5:Pawn
        self.board = board
        
    def getBoard(self):
        return self.board
    
    def clone(self):
        new_game = MicroChess()
        new_game.setBoard(self.board.copy())
        new_game.player = self.player
        new_game.winner = self.winner
        new_game.done = self.done
        new_game.moves = self.moves
        return new_game
    
    def showBoard(self):
        print(self.board)
        
    def isPathClear(self, fromX, fromY, toX, toY):
        stepX = 0 if fromX == toX else (toX - fromX) // abs(toX - fromX)
        stepY = 0 if fromY == toY else (toY - fromY) // abs(toY - fromY)
        currentX, currentY = fromX + stepX, fromY + stepY

        while currentX != toX or currentY != toY:
            if self.board[currentX][currentY] != 0:
                return False
            currentX += stepX
            currentY += stepY

        return True
        
    def getKingValidMoves(self, x, y):
        validMoves = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i == 0 and j == 0):
                    continue
                if (x+i >= 0 and x+i < 5 and y+j >= 0 and y+j < 4):
                    if (self.board[x+i][y+j] * self.player <= 0):
                        validMoves.append((x+i, y+j))
        return validMoves
    
    def getKnightValidMoves(self, x, y):
        validMoves = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (abs(i) == 2 and abs(j) == 1) or (abs(i) == 1 and abs(j) == 2):
                    if (x+i >= 0 and x+i < 5 and y+j >= 0 and y+j < 4):
                        if (self.board[x+i][y+j] * self.player <= 0):
                            validMoves.append((x+i, y+j))
        return validMoves
    
    def getBishopValidMoves(self, x, y):
        validMoves = []
        for i in range(-4, 5):
            for j in range(-4, 4):
                if abs(i) == abs(j) and i != 0:  # 斜め方向のみ
                    if x+i >= 0 and x+i < 5 and y+j >= 0 and y+j < 4:
                        if self.board[x+i][y+j] * self.player <= 0 and self.isPathClear(x, y, x+i, y+j):
                            validMoves.append((x+i, y+j))
        return validMoves
    
    def getRookValidMoves(self, x, y):
        validMoves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 上下左右の方向
        
        for dx, dy in directions:
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < 5 and 0 <= ny < 4:
                    if self.board[nx][ny] * self.player <= 0:
                        if self.isPathClear(x, y, nx, ny):
                            validMoves.append((nx, ny))
                        if self.board[nx][ny] != 0:  # 敵の駒がある場合、そこでストップ
                            break
                    else:
                        break
        return validMoves
    
    def getPawnValidMoves(self, x, y):
        validMoves = []
        direction = -1 if self.player == 1 else 1  # 白は上 (-1), 黒は下 (+1)

        # 前方の移動
        if 0 <= x + direction < 5 and self.board[x + direction][y] == 0:
            validMoves.append((x + direction, y))

        # 斜めの攻撃
        for dy in [-1, 1]:
            if 0 <= y + dy < 4 and 0 <= x + direction < 5:
                if self.board[x + direction][y + dy] * self.player < 0:
                    validMoves.append((x + direction, y + dy))
        return validMoves

    def getValidMoves(self, x, y):
        piece = self.board[x][y] * self.player
        if (piece == 1):
            return self.getKingValidMoves(x, y)
        elif (piece == 2):
            return self.getKnightValidMoves(x, y)
        elif (piece == 3):
            return self.getBishopValidMoves(x, y)
        elif (piece == 4):
            return self.getRookValidMoves(x, y)
        elif (piece == 5):
            return self.getPawnValidMoves(x, y)
        return []
    
    def isInCheck(self, player):
        king_piece = 1 if player == 1 else -1
        king_position = None
        # キングの位置を探す
        for x in range(5):
            for y in range(4):
                if self.board[x][y] == king_piece:
                    king_position = (x, y)
                    break
            if king_position:
                break
        if not king_position:
            return False  # キングが盤上にない場合（ゲームは既に終了しているはず）

        opponent = -player
        # 相手の全ての駒がキングの位置に移動できるかを確認
        for x in range(5):
            for y in range(4):
                if self.board[x][y] * opponent > 0:
                    valid_moves = self.getValidMoves(x, y)
                    if king_position in valid_moves:
                        return True
        return False
    
    def isInsufficientMaterial(self):
        # 各プレイヤーの駒の種類をカウント
        white_pieces = self.board[self.board > 0]
        black_pieces = self.board[self.board < 0]

        # 白と黒の駒がキングのみか確認
        white_only_king = np.array_equal(white_pieces, [1]) if len(white_pieces) == 1 else False
        black_only_king = np.array_equal(black_pieces, [-1]) if len(black_pieces) == 1 else False

        return white_only_king and black_only_king

    
    def checkWinner(self):
        # キングが盤上に存在するか確認
        white_king = np.any(self.board == 1)
        black_king = np.any(self.board == -1)

        if not black_king:
            self.winner = 1
            self.done = True
            return self.winner
        elif not white_king:
            self.winner = -1
            self.done = True
            return self.winner

        # 現在のプレイヤーがチェックされているか確認
        if self.isInCheck(self.player):
            if not self.getValidMovesList():
                self.winner = -self.player
                self.done = True
                return self.winner

        # ステイルメイトの確認：チェックされていないが、合法的な手がない場合
        if not self.getValidMovesList():
            self.winner = 0  # 引き分け
            self.done = True
            return self.winner

        # 不十分な駒による引き分けの確認
        if self.isInsufficientMaterial():
            self.winner = 0  # 引き分け
            self.done = True
            return self.winner
        
        if self.moves >= 50:
            self.winner = 0
            self.done = True

        return self.winner

    
    def reset(self):
        self.board = np.array(((-1, -2, -3, -4), (-5, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 5), (4, 3, 2, 1)), dtype=int)
        self.player = 1
        self.winner = 0
        self.done = False
        self.moves = 0
        
    def getValidMovesList(self):
        validMovesList = []
        for i in range(5):
            for j in range(4):
                if (self.board[i][j] * self.player > 0):
                    validMoves = self.getValidMoves(i, j)
                    for move in validMoves:
                        validMovesList.append((i, j, move[0], move[1]))
        return validMovesList
    
    def move(self, x, y, x1, y1):
        if (self.done):
            return
        if (self.board[x][y] * self.player <= 0):
            return
        validMoves = self.getValidMoves(x, y)
        if ((x1, y1) in validMoves):
            self.board[x1][y1] = self.board[x][y]
            self.board[x][y] = 0
            self.player *= -1
            self.moves += 1
            self.checkWinner()
            if (self.moves >= 50):
                self.done = True
        else:
            return
        
    def randomMove(self):
        validMovesList = self.getValidMovesList()
        if (len(validMovesList) == 0):
            self.done = True
            return
        move = validMovesList[np.random.randint(len(validMovesList))]
        self.move(move[0], move[1], move[2], move[3])
        