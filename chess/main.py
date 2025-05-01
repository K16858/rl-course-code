from MicroChess import MicroChess

chess = MicroChess()
chess.setBoard()

while(not chess.done):
    chess.showBoard()
    chess.randomMove()

chess.showBoard()
print("Winner: ", chess.winner, chess.moves)
