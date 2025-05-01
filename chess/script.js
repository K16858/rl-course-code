document.addEventListener("DOMContentLoaded", () => {
    createBoard();
});

function createBoard() {
    const board = document.getElementById("board");
    const rows = 5;
    const cols = 4;
    
    // 初期駒の配置
    const pieces = {
        "0-0": "bK.png", "0-1": "bN.png", "0-2": "bB.png", "0-3": "bR.png",
        "1-0": "bP.png",
        "3-3": "wP.png",
        "4-0": "wR.png", "4-1": "wB.png", "4-2": "wN.png", "4-3": "wK.png"
    };

    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            let square = document.createElement("div");
            square.classList.add("square");
            square.classList.add((row + col) % 2 === 0 ? "white" : "black");
            square.dataset.position = `${row}-${col}`;

            square.addEventListener("dragover", (e) => e.preventDefault());
            square.addEventListener("drop", handleDrop);

            const pieceKey = `${row}-${col}`;
            if (pieces[pieceKey]) {
                let piece = document.createElement("img");
                piece.classList.add("piece");
                piece.src = `img/${pieces[pieceKey]}`;
                piece.alt = pieces[pieceKey].split('.')[0];
                piece.draggable = true;
                piece.addEventListener("dragstart", handleDragStart);
                square.appendChild(piece);
            }

            board.appendChild(square);
        }
    }
}

let draggedPiece = null;
let dragStartPosition = null;
let isPlayerTurn = true; // プレイヤーが白（true）、AIが黒（false）

// 盤面を2次元配列で表現
function getBoardState() {
    const state = Array(5).fill().map(() => Array(4).fill(0));
    const squares = document.querySelectorAll('.square');
    
    squares.forEach(square => {
        const [row, col] = square.dataset.position.split('-').map(Number);
        const piece = square.querySelector('img');
        if (piece) {
            const pieceType = piece.alt;
            let value = 0;
            // 駒の種類と色に基づいて値を設定
            if (pieceType.startsWith('w')) { // 白の駒
                if (pieceType.endsWith('K')) value = 1;
                else if (pieceType.endsWith('N')) value = 2;
                else if (pieceType.endsWith('B')) value = 3;
                else if (pieceType.endsWith('R')) value = 4;
                else if (pieceType.endsWith('P')) value = 5;
            } else { // 黒の駒
                if (pieceType.endsWith('K')) value = -1;
                else if (pieceType.endsWith('N')) value = -2;
                else if (pieceType.endsWith('B')) value = -3;
                else if (pieceType.endsWith('R')) value = -4;
                else if (pieceType.endsWith('P')) value = -5;
            }
            state[row][col] = value;
        }
    });
    return state;
}

// AIの手を取得
async function getAIMove(state) {
    try {
        const response = await fetch('http://localhost:8000/move/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                state_3d: [state], // state_3dとして3D配列を送信
                valid_moves: getValidMoves(state)
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        return data.best_move;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// 有効な手のリストを生成（簡易版）
function getValidMoves(state) {
    const moves = [];
    // すべての可能な手を生成
    for (let fromRow = 0; fromRow < 5; fromRow++) {
        for (let fromCol = 0; fromCol < 4; fromCol++) {
            for (let toRow = 0; toRow < 5; toRow++) {
                for (let toCol = 0; toCol < 4; toCol++) {
                    if (fromRow !== toRow || fromCol !== toCol) {
                        moves.push([fromRow, fromCol, toRow, toCol]);
                    }
                }
            }
        }
    }
    return moves;
}

// AIの手を実行
function makeAIMove(move) {
    const [fromRow, fromCol, toRow, toCol] = move;
    const fromSquare = document.querySelector(`[data-position="${fromRow}-${fromCol}"]`);
    const toSquare = document.querySelector(`[data-position="${toRow}-${toCol}"]`);
    const piece = fromSquare.querySelector('img');
    
    if (piece) {
        // 移動先に駒がある場合は取る
        if (toSquare.querySelector('img')) {
            toSquare.removeChild(toSquare.querySelector('img'));
        }
        toSquare.appendChild(piece);
        
        // AIの手の実行後、明示的にプレイヤーのターンに設定
        setTimeout(() => {
            isPlayerTurn = true;
            console.log("プレイヤーのターンに切り替わりました");
        }, 100);
    } else {
        console.error("AIの手の実行に失敗しました：駒が見つかりません");
        console.error(piece, fromRow, fromCol, toRow, toCol);
        isPlayerTurn = true; // エラー時もプレイヤーのターンに戻す
    }
}

function handleDragStart(e) {
    if (!isPlayerTurn) return; // プレイヤーのターンでない場合はドラッグ不可
    
    draggedPiece = e.target;
    dragStartPosition = e.target.parentElement.dataset.position;
    setTimeout(() => {
        draggedPiece.style.visibility = "hidden";
    }, 0);
}

async function handleDrop(e) {
    e.preventDefault();
    if (!draggedPiece || !isPlayerTurn) return;

    const targetSquare = e.target.classList.contains('square') ? e.target : e.target.parentElement;
    draggedPiece.style.visibility = "visible";

    // 移動先に駒がある場合は取る
    if (targetSquare.querySelector('img')) {
        targetSquare.removeChild(targetSquare.querySelector('img'));
    }
    targetSquare.appendChild(draggedPiece);
    
    // プレイヤーの手を完了し、AIのターンへ
    isPlayerTurn = false;
    draggedPiece = null;

    // 盤面の状態を取得してAIの手を要求
    const state = getBoardState();
    const aiMove = await getAIMove(state);
    console.log(state);
    if (aiMove) {
        makeAIMove(aiMove); // 少し遅延を入れてAIの手を実行
    }
}
