import sys
import time


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


def parse_setoption(parts):
    try:
        name_start = parts.index("name") + 1
        value_start = parts.index("value", name_start)
    except ValueError:
        return None, None

    name = " ".join(parts[name_start:value_start]).strip().lower()
    value = " ".join(parts[value_start + 1 :])
    return name, value


def parse_check_option(value):
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    return None


def allocate_time(time_left, inc):
    if time_left <= 0:
        return 1.0
    if time_left < 2.0:
        return 0.05
    if time_left < 5.0:
        return 0.15
    return max(0.08, (time_left - 0.5) / 25.0 + inc * 0.75)


def handle_position(board, parts, chess):
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


def warmup(searcher, chess):
    print("info string Warming up Numba JIT...", flush=True)
    start = time.perf_counter()
    try:
        dummy_board = chess.Board()
        searcher.search(dummy_board, fixed_depth=4, shut_up=True)
        searcher.search(dummy_board, fixed_depth=4, shut_up=True)
        searcher.tt.clear()
        searcher.history.clear()
        searcher.counter_moves.clear()
        elapsed = (time.perf_counter() - start) * 1000
        print(
            f"info string Numba JIT warmup complete in {int(elapsed)} ms!", flush=True
        )
    except Exception as e:
        print(f"info string Warmup failed: {e}", flush=True)


def uci_loop(model_path="guitarfish.gm", book_path="", tt_size_mb=256):
    evaluator = None
    searcher = None
    chess = None
    board = None
    own_book = True
    syzygy_path = ""
    online_tablebase_enabled = True

    def ensure_model_and_numba_warmed_up():
        nonlocal board, chess, evaluator, searcher
        if searcher is None:
            import src.bitboard as chess
            from src.evaluate import Evaluator
            from src.search import Searcher

            new_board = chess.Board()
            new_evaluator = Evaluator(model_path)
            new_searcher = Searcher(
                new_evaluator, book_path, tt_size_mb=tt_size_mb
            )
            try:
                new_searcher.own_book = own_book
                new_searcher.online_tablebase_enabled = online_tablebase_enabled
                if syzygy_path:
                    new_searcher.set_syzygy_path(syzygy_path)
                warmup(new_searcher, chess)
            except Exception:
                new_searcher.close()
                raise

            board = new_board
            evaluator = new_evaluator
            searcher = new_searcher

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            print("id name Guitarfish v6.4")
            print("id author memeviber")
            print(f"option name EvalFile type string default {model_path}")
            print(f"option name BookFile type string default {book_path}")
            print(f"option name Hash type spin default {tt_size_mb} min 1 max 1024")
            print("option name OwnBook type check default true")
            print("option name SyzygyPath type string default")
            print("option name OnlineTablebase type check default true")
            print("uciok", flush=True)

        elif cmd == "isready":
            try:
                ensure_model_and_numba_warmed_up()
            except Exception as exc:
                print(f"info string Engine initialization failed: {exc}", flush=True)
            else:
                print("readyok", flush=True)

        elif cmd == "setoption":
            name, value = parse_setoption(parts)
            if name == "evalfile":
                if searcher is None:
                    model_path = value
                else:
                    from src.evaluate import Evaluator

                    try:
                        updated_evaluator = Evaluator(value)
                    except Exception as exc:
                        print(f"info string EvalFile load failed: {exc}", flush=True)
                    else:
                        evaluator = updated_evaluator
                        searcher.evaluator = updated_evaluator
                        model_path = value
            elif name == "bookfile":
                if searcher is None:
                    book_path = value
                else:
                    try:
                        searcher.book = chess.polyglot.open_reader(value)
                    except Exception as exc:
                        print(f"info string BookFile update failed: {exc}", flush=True)
                    else:
                        book_path = value
            elif name == "hash":
                try:
                    size_mb = int(value)
                    if not 1 <= size_mb <= 1024:
                        raise ValueError
                except ValueError:
                    print(
                        "info string Hash must be an integer from 1 to 1024", flush=True
                    )
                else:
                    tt_size_mb = size_mb
                    if searcher is not None:
                        searcher.set_hash_size(size_mb)
            elif name == "ownbook":
                enabled = parse_check_option(value)
                if enabled is None:
                    print("info string OwnBook expects true or false", flush=True)
                else:
                    own_book = enabled
                    if searcher is not None:
                        searcher.own_book = enabled
            elif name == "syzygypath":
                if searcher is None:
                    syzygy_path = value
                else:
                    try:
                        searcher.set_syzygy_path(value)
                    except Exception as exc:
                        print(f"info string SyzygyPath update failed: {exc}", flush=True)
                    else:
                        syzygy_path = value
            elif name == "onlinetablebase":
                enabled = parse_check_option(value)
                if enabled is None:
                    print(
                        "info string OnlineTablebase expects true or false", flush=True
                    )
                else:
                    online_tablebase_enabled = enabled
                    if searcher is not None:
                        searcher.online_tablebase_enabled = enabled

        elif cmd == "ucinewgame":
            if board is not None:
                board.reset()
            if searcher is not None:
                searcher.tt.clear()
                searcher.history.clear()
                searcher.counter_moves.clear()

        elif cmd == "position":
            if board is None:
                print(
                    "info string Engine is not initialized; send isready first",
                    flush=True,
                )
            else:
                handle_position(board, parts, chess)

        elif cmd == "go":
            if searcher is None or board is None:
                print(
                    "info string Engine is not initialized; send isready first",
                    flush=True,
                )
                continue

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
                searcher.search(board, fixed_depth=args["depth"], time_limit=99999999.0)
            elif "movetime" in args:
                searcher.search(board, time_limit=args["movetime"] / 1000.0)
            else:
                alloc = allocate_time(time_left, inc)
                searcher.search(board, time_limit=alloc)

            sys.stdout.flush()

        elif cmd == "quit":
            break

    if searcher is not None:
        searcher.close()
