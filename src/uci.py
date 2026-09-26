import sys

import src.bitboard as chess
from src.evaluate import Evaluator
from src.search import Searcher


def uci_loop(model_path="guitarfish.gm", book_path="book.bin"):
    evaluator = Evaluator(model_path)
    searcher = Searcher(evaluator, book_path)
    board = chess.Board()

    author = evaluator.metadata.get("author", "MemeViber")
    desc = evaluator.metadata.get("description", "5.3.28@6.4")

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
        except EOFError:
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            print(f"id name Guitarfish [{desc}]")
            print(f"id author {author}")
            print("uciok")
            sys.stdout.flush()
        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()
        elif cmd == "ucinewgame":
            board.reset()
            searcher.tt.clear()
            searcher.history.clear()
            searcher.counter_moves.clear()
        elif cmd == "position":
            idx = parts.index("moves") if "moves" in parts else -1
            if parts[1] == "startpos":
                board.reset()
                moves = parts[idx + 1 :] if idx != -1 else []
            elif parts[1] == "fen":
                fen = " ".join(parts[2:idx]) if idx != -1 else " ".join(parts[2:])
                board.set_fen(fen)
                moves = parts[idx + 1 :] if idx != -1 else []
            for m in moves:
                board.push(chess.Move.from_uci(m))
        elif cmd == "go":
            wtime = btime = 0
            winc = binc = 0
            movetime = None
            depth = None

            if "wtime" in parts:
                wtime = int(parts[parts.index("wtime") + 1])
            if "btime" in parts:
                btime = int(parts[parts.index("btime") + 1])
            if "winc" in parts:
                winc = int(parts[parts.index("winc") + 1])
            if "binc" in parts:
                binc = int(parts[parts.index("binc") + 1])
            if "movetime" in parts:
                movetime = int(parts[parts.index("movetime") + 1])
            if "depth" in parts:
                depth = int(parts[parts.index("depth") + 1])

            time_left = (wtime if board.turn == chess.WHITE else btime) / 1000.0
            inc = (winc if board.turn == chess.WHITE else binc) / 1000.0

            last_opp_move = (
                board.move_stack[-1].uci() if len(board.move_stack) > 0 else None
            )
            if (
                time_left > 0
                and time_left < 3.0
                and searcher.expected_opponent_move == last_opp_move
                and searcher.premove_reply is not None
            ):
                premove = chess.Move.from_uci(searcher.premove_reply)
                if board.is_legal(premove):
                    print(f"bestmove {searcher.premove_reply}")
                    sys.stdout.flush()
                    continue

            if depth:
                searcher.search(board, time_limit=9999, fixed_depth=depth)
            elif movetime:
                searcher.search(board, time_limit=movetime / 1000.0)
            else:
                if time_left <= 0:
                    alloc = 1.0
                elif time_left < 2.0:
                    alloc = 0.05
                elif time_left < 5.0:
                    alloc = 0.15
                else:
                    alloc = max(0.08, (time_left - 0.5) / 25.0 + inc * 0.75)
                searcher.search(board, time_limit=alloc)
            sys.stdout.flush()
        elif cmd == "quit":
            break
