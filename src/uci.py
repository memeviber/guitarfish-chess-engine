import sys

import src.bitboard as chess
from src.evaluate import Evaluator
from src.search import Searcher


def parse_go(parts):
    args = {}
    it = iter(parts[1:])
    for token in it:
        if token in ("wtime", "btime", "winc", "binc", "movetime", "depth"):
            try:
                args[token] = int(next(it))
            except (StopIteration, ValueError):
                pass
    return args


def allocate_time(time_left, inc):
    if time_left <= 0:
        return 1.0
    if time_left < 2.0:
        return 0.05
    if time_left < 5.0:
        return 0.15
    return max(0.08, (time_left - 0.5) / 25.0 + inc * 0.75)


def handle_position(board, parts):
    try:
        idx = parts.index("moves")
    except ValueError:
        idx = -1

    if parts[1] == "startpos":
        board.reset()
        moves = parts[idx + 1 :] if idx != -1 else []
    elif parts[1] == "fen":
        fen = " ".join(parts[2:idx]) if idx != -1 else " ".join(parts[2:])
        board.set_fen(fen)
        moves = parts[idx + 1 :] if idx != -1 else []
    else:
        return

    for m in moves:
        board.push(chess.Move.from_uci(m))


def warmup(searcher):
    print("info string Warming up Numba JIT...", flush=True)
    dummy_board = chess.Board()
    searcher.search(dummy_board, fixed_depth=4, shut_up=True)
    searcher.search(dummy_board, fixed_depth=4, shut_up=True)
    searcher.tt.clear()
    searcher.history.clear()
    searcher.counter_moves.clear()
    print("info string Numba JIT warmup complete!", flush=True)


def uci_loop(model_path="guitarfish.gm", book_path="book.bin"):
    evaluator = Evaluator(model_path)
    searcher = Searcher(evaluator, book_path)
    board = chess.Board()

    warmup(searcher)

    author = evaluator.metadata.get("author", "MemeViber")
    desc = evaluator.metadata.get("description", "6.1.45@6.4")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            print(f"id name Guitarfish [{desc}]")
            print(f"id author {author}")
            print("uciok", flush=True)

        elif cmd == "isready":
            print("readyok", flush=True)

        elif cmd == "ucinewgame":
            board.reset()
            searcher.tt.clear()
            searcher.history.clear()
            searcher.counter_moves.clear()

        elif cmd == "position":
            handle_position(board, parts)

        elif cmd == "go":
            args = parse_go(parts)
            is_white = board.turn == chess.WHITE

            wtime = args.get("wtime", 0)
            btime = args.get("btime", 0)
            winc = args.get("winc", 0)
            binc = args.get("binc", 0)

            time_left = (wtime if is_white else btime) / 1000.0
            inc = (winc if is_white else binc) / 1000.0

            last_opp_move = board.move_stack[-1].uci() if board.move_stack else None
            if (
                0 < time_left < 3.0
                and searcher.expected_opponent_move == last_opp_move
                and searcher.premove_reply is not None
            ):
                premove = chess.Move.from_uci(searcher.premove_reply)
                if board.is_legal(premove):
                    print(f"bestmove {searcher.premove_reply}", flush=True)
                    continue

            if "depth" in args:
                searcher.search(board, fixed_depth=args["depth"])
            elif "movetime" in args:
                searcher.search(board, time_limit=args["movetime"] / 1000.0)
            else:
                alloc = allocate_time(time_left, inc)
                searcher.search(board, time_limit=alloc)

            sys.stdout.flush()

        elif cmd == "quit":
            break
