#!/usr/bin/env python3
import io
import json
import math
import os
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort
from numba import njit

from src import bitboard as chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

MATE_VALUE = 30000
MAX_PLY = 64
DEFAULT_WDL_SCALE = 410.0

LMR_TABLE = [[0 for _ in range(64)] for _ in range(64)]
for d in range(1, 64):
    for mc in range(1, 64):
        LMR_TABLE[d][mc] = int(0.75 + math.log(d) * math.log(mc) / 2.25)


@njit(cache=True, fastmath=True)
def accumulate_features(out_acc, ft_w, ft_b, indices, count):
    for j in range(ft_b.shape[0]):
        out_acc[j] = ft_b[j]
    for i in range(count):
        row = ft_w[indices[i]]
        for j in range(ft_b.shape[0]):
            out_acc[j] += row[j]


def resolve_resource_path(resource_path):
    if os.path.exists(resource_path):
        return resource_path
    if hasattr(sys, "_MEIPASS"):
        bundled_path = os.path.join(sys._MEIPASS, resource_path)
        if os.path.exists(bundled_path):
            return bundled_path
        default_bundled = os.path.join(sys._MEIPASS, os.path.basename(resource_path))
        if os.path.exists(default_bundled):
            return default_bundled
    return resource_path


class Evaluator:
    def __init__(self, model_path="guitarfish_v2.gm"):
        model_path = resolve_resource_path(model_path)
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.metadata = {}
        self.wdl_scale = DEFAULT_WDL_SCALE

        if model_path.endswith(".gm") or model_path.endswith(".zip"):
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            with zipfile.ZipFile(model_path, "r") as z:
                if "metadata.json" in z.namelist():
                    self.metadata = json.loads(z.read("metadata.json").decode("utf-8"))
                    self.wdl_scale = float(
                        self.metadata.get("wdl_scale", DEFAULT_WDL_SCALE)
                    )

                npz_bytes = z.read("weights.npz")
                with np.load(io.BytesIO(npz_bytes)) as w:
                    self.ft_w = np.ascontiguousarray(w["ft_weight"], dtype=np.float32)
                    self.ft_b = np.ascontiguousarray(w["ft_bias"], dtype=np.float32)

                onnx_bytes = z.read("mlp_int8.onnx")
                self.session = ort.InferenceSession(
                    onnx_bytes, sess_options=options, providers=["CPUExecutionProvider"]
                )
        else:
            feature_path = "guitarfish_weights.npz"
            onnx_path = (
                model_path
                if model_path.endswith(".onnx")
                else "guitarfish_mlp_int8.onnx"
            )
            with np.load(feature_path) as w:
                self.ft_w = np.ascontiguousarray(w["ft_weight"], dtype=np.float32)
                self.ft_b = np.ascontiguousarray(w["ft_bias"], dtype=np.float32)
            self.session = ort.InferenceSession(
                onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
            )

        self.l1_size = self.ft_b.shape[0]
        self.buf_stm_indices = np.empty(256, dtype=np.int64)
        self.buf_opp_indices = np.empty(256, dtype=np.int64)
        self.acc_stm = np.empty(self.l1_size, dtype=np.float32)
        self.acc_opp = np.empty(self.l1_size, dtype=np.float32)
        self.onnx_input = np.empty((1, self.l1_size * 2), dtype=np.float32)

        self._warmup()

    def _warmup(self):
        for _ in range(3):
            chess.Board().perft(4)
            self.session.run(
                None,
                {"features": np.zeros((1, self.l1_size * 2), dtype=np.float32)},
            )

    def evaluate(self, board: chess.Board):
        stm_cnt, opp_cnt = chess.extract_indices(
            board._pieces,
            board.turn,
            self.buf_stm_indices,
            self.buf_opp_indices,
        )
        accumulate_features(
            self.acc_stm,
            self.ft_w,
            self.ft_b,
            self.buf_stm_indices,
            stm_cnt,
        )
        accumulate_features(
            self.acc_opp,
            self.ft_w,
            self.ft_b,
            self.buf_opp_indices,
            opp_cnt,
        )

        self.onnx_input[0, : self.l1_size] = self.acc_stm
        self.onnx_input[0, self.l1_size :] = self.acc_opp

        logit = self.session.run(None, {"features": self.onnx_input})[0][0, 0]
        return int(logit * np.float32(self.wdl_scale))


class TranspositionTable:
    def __init__(self, size_mb=256):
        num_entries = (size_mb * 1024 * 1024) // 32
        self.size = 1 << (num_entries.bit_length() - 1)
        self.mask = self.size - 1
        self.table = [None] * self.size

    def store(self, zobrist, depth, flag, score, move, ply):
        if score > MATE_VALUE - 1000:
            score += ply
        elif score < -MATE_VALUE + 1000:
            score -= ply

        idx = zobrist & self.mask
        entry = self.table[idx]
        if entry is None or depth >= entry[1] or flag == 0:
            self.table[idx] = (zobrist, depth, flag, score, move)

    def probe(self, zobrist, depth, alpha, beta, ply):
        entry = self.table[zobrist & self.mask]
        if entry and entry[0] == zobrist:
            _, e_depth, flag, score, move = entry
            if score > MATE_VALUE - 1000:
                score -= ply
            elif score < -MATE_VALUE + 1000:
                score += ply

            if e_depth >= depth:
                if flag == 0:
                    return score
                if flag == 1 and score <= alpha:
                    return alpha
                if flag == 2 and score >= beta:
                    return beta
            return move
        return None

    def get(self, zobrist):
        entry = self.table[zobrist & self.mask]
        if entry and entry[0] == zobrist:
            return entry
        return None

    def clear(self):
        self.table = [None] * self.size


class Searcher:
    def __init__(self, evaluator, book_path="book.bin"):
        self.evaluator = evaluator
        book_path = resolve_resource_path(book_path)
        self.book = chess.polyglot.open_reader(book_path)
        self.tt = TranspositionTable(size_mb=256)
        self.history = {}
        self.killers = {}
        self.counter_moves = {}
        self.eval_stack = [0] * MAX_PLY
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 0
        self.stop = False
        self.sel_depth = 0
        self.expected_opponent_move = None
        self.premove_reply = None
        self.pv_lines = [[] for _ in range(MAX_PLY)]

    def score_move(self, board, move, depth, tt_move=None, prev_move=None):
        if move == tt_move:
            return 2_000_000

        if board.is_capture(move):
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            if chess.fast_see(
                board._pieces, board.turn, move.from_square, move.to_square
            ):
                return 1_000_000 + (PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker])
            else:
                return -100_000 + (PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker])

        k = self.killers.get(depth)
        if k:
            if move == k[0]:
                return 900_000
            if move == k[1]:
                return 800_000

        if (
            prev_move
            and (prev_move in self.counter_moves)
            and move == self.counter_moves[prev_move]
        ):
            return 750_000

        return self.history.get((board.turn, move.from_square, move.to_square), 0)

    def quiescence(self, board, alpha, beta, ply=0):
        self.nodes += 1
        if (self.nodes & 2047) == 0 and time.time() - self.start_time > self.time_limit:
            self.stop = True
        if self.stop:
            return 0

        if ply > self.sel_depth:
            self.sel_depth = ply

        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        if stand_pat < alpha - 900:
            return alpha

        captures = list(board.generate_legal_captures())
        if not captures:
            return alpha

        scored_captures = []
        for m in captures:
            see_val = board.see(m)
            if see_val < 0:
                continue

            victim = board.piece_type_at(m.to_square) or chess.PAWN
            attacker = board.piece_type_at(m.from_square) or chess.PAWN
            mvv_lva = PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker]
            scored_captures.append((see_val * 100 + mvv_lva, m))

        if not scored_captures:
            return alpha

        scored_captures.sort(key=lambda x: x[0], reverse=True)

        for _, move in scored_captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()

            if self.stop:
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board, depth, alpha, beta, ply, allow_nmp=True, prev_move=None):
        self.nodes += 1
        if ply < MAX_PLY:
            self.pv_lines[ply] = []
        if ply > 0 and (board.is_repetition(2) or board.can_claim_fifty_moves()):
            return 0

        if (self.nodes & 2047) == 0 and time.time() - self.start_time > self.time_limit:
            self.stop = True
        if self.stop:
            return 0

        if ply >= MAX_PLY - 1:
            return self.evaluator.evaluate(board)

        zobrist = chess.polyglot.zobrist_hash(board)
        tt_data = self.tt.probe(zobrist, depth, alpha, beta, ply)
        tt_move = None
        if tt_data is not None:
            if isinstance(tt_data, int):
                if ply > 0:
                    return tt_data
            else:
                tt_move = tt_data

        in_check = board.is_check()
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        if not in_check:
            static_eval = self.evaluator.evaluate(board)
            self.eval_stack[ply] = static_eval
            improving = ply >= 2 and static_eval > self.eval_stack[ply - 2]
        else:
            static_eval = -MATE_VALUE
            self.eval_stack[ply] = static_eval
            improving = False

        if not in_check and depth <= 6 and abs(beta) < MATE_VALUE - 1000:
            rfp_margin = (80 - 20 * improving) * depth
            if static_eval - rfp_margin >= beta:
                return static_eval - rfp_margin

        if (
            allow_nmp
            and not in_check
            and ply > 0
            and depth >= 3
            and static_eval >= beta
        ):
            if board.piece_count() > 5:
                null_move = chess.Move.null()
                board.push(null_move)
                R = 3 + (depth // 4) + (1 if improving else 0)
                null_score = -self.negamax(
                    board,
                    depth - 1 - R,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    False,
                    None,
                )
                board.pop()

                if self.stop:
                    return 0
                if null_score >= beta:
                    return beta

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return -MATE_VALUE + ply if in_check else 0

        legal_moves.sort(
            key=lambda m: self.score_move(board, m, depth, tt_move, prev_move),
            reverse=True,
        )

        best_score = -MATE_VALUE
        best_move_found = None
        moves_searched = 0
        alpha_orig = alpha
        quiets_searched = []

        for move in legal_moves:
            is_cap = board.is_capture(move)
            board.push(move)
            moves_searched += 1

            extension = 1 if (board.is_check() and ply < 32) else 0
            new_depth = depth - 1 + extension

            if moves_searched == 1:
                score = -self.negamax(
                    board, new_depth, -beta, -alpha, ply + 1, True, move
                )
            else:
                r = 0
                if depth >= 3 and moves_searched > 1 and not is_cap:
                    d_idx = min(depth, 63)
                    m_idx = min(moves_searched, 63)
                    r = LMR_TABLE[d_idx][m_idx]
                    if improving:
                        r -= 1
                    h_val = self.history.get(
                        (board.turn, move.from_square, move.to_square), 0
                    )
                    if h_val > 5000:
                        r -= 1
                    r = max(0, min(r, new_depth - 1))

                score = -self.negamax(
                    board,
                    new_depth - r,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    True,
                    move,
                )
                if score > alpha and (score < beta or r > 0):
                    score = -self.negamax(
                        board, new_depth, -beta, -alpha, ply + 1, True, move
                    )

            board.pop()

            if self.stop:
                return 0

            if not is_cap:
                quiets_searched.append(move)

            if score > best_score:
                best_score = score
                best_move_found = move
                if ply < MAX_PLY - 1:
                    self.pv_lines[ply] = [move] + self.pv_lines[ply + 1]

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not is_cap:
                    k = self.killers.setdefault(depth, [None, None])
                    if move != k[0]:
                        k[1] = k[0]
                        k[0] = move
                    bonus = depth * depth
                    key_best = (board.turn, move.from_square, move.to_square)
                    self.history[key_best] = self.history.get(key_best, 0) + bonus

                    if prev_move:
                        self.counter_moves[prev_move] = move

                    for qm in quiets_searched:
                        if qm != move:
                            key_bad = (
                                board.turn,
                                qm.from_square,
                                qm.to_square,
                            )
                            self.history[key_bad] = self.history.get(key_bad, 0) - bonus
                break

        flag = 0
        if best_score <= alpha_orig:
            flag = 1
        elif best_score >= beta:
            flag = 2

        self.tt.store(zobrist, depth, flag, best_score, best_move_found, ply)
        return best_score

    def get_pv_from_tt(self, board, max_length=64):
        pv = []
        curr = board.copy()
        seen = set()

        for _ in range(max_length):
            zobrist = chess.polyglot.zobrist_hash(curr)
            if zobrist in seen:
                break
            seen.add(zobrist)

            entry = self.tt.get(zobrist)
            if not entry:
                break

            move = entry[4]
            if not move or not curr.is_legal(move):
                break

            pv.append(move)
            curr.push(move)

        return pv

    def search(self, board, time_limit=5, fixed_depth=None):
        if not fixed_depth or fixed_depth > 1:
            book_move = self.book.probe(board)
            if book_move:
                print(f"info depth 1 score cp 20 time 1 pv {book_move.uci()}")
                print(f"bestmove {book_move.uci()}")
                return

        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.stop = False
        self.sel_depth = 0
        self.killers.clear()

        for k in self.history:
            self.history[k] = self.history[k] * 3 // 4

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("bestmove (none)")
            return

        best_move = legal_moves[0].uci()
        ponder_move = None
        best_val_prev = None
        max_depth = fixed_depth if fixed_depth else 64

        for depth in range(1, max_depth + 1):
            self.sel_depth = 0

            if depth >= 5 and best_val_prev is not None:
                delta = 35
                alpha = max(-MATE_VALUE, best_val_prev - delta)
                beta = min(MATE_VALUE, best_val_prev + delta)

                while True:
                    score = self.negamax(board, depth, alpha, beta, 0)
                    if self.stop:
                        break

                    if score <= alpha:
                        alpha = max(-MATE_VALUE, alpha - delta)
                    elif score >= beta:
                        beta = min(MATE_VALUE, beta + delta)
                    else:
                        break

                    delta += delta // 2
                    if delta > 1000:
                        score = self.negamax(board, depth, -MATE_VALUE, MATE_VALUE, 0)
                        break
            else:
                score = self.negamax(board, depth, -MATE_VALUE, MATE_VALUE, 0)

            if self.stop:
                break

            best_val_prev = score

            pv_list = self.get_pv_from_tt(board)
            if not pv_list and self.pv_lines[0]:
                pv_list = self.pv_lines[0]

            if pv_list:
                pv_str = " ".join(m.uci() for m in pv_list)
                best_move = pv_list[0].uci()
                if len(pv_list) > 1:
                    ponder_move = pv_list[1].uci()
                else:
                    ponder_move = None

                if len(pv_list) > 2:
                    self.expected_opponent_move = ponder_move
                    self.premove_reply = pv_list[2].uci()
                else:
                    self.expected_opponent_move = None
                    self.premove_reply = None
            else:
                pv_str = best_move

            elapsed = max(time.time() - self.start_time, 1e-6)
            nps = int(self.nodes / elapsed)
            score_str = (
                f"mate {(MATE_VALUE - abs(score) + 1) // 2 if score > 0 else -((MATE_VALUE - abs(score) + 1) // 2)}"
                if abs(score) > MATE_VALUE - 1000
                else f"cp {score}"
            )

            print(
                f"info depth {depth} seldepth {self.sel_depth} score {score_str} "
                f"nodes {self.nodes} nps {nps} time {int(elapsed * 1000)} pv {pv_str}"
            )

            if not fixed_depth and elapsed > self.time_limit * 0.6:
                break

        if best_move and ponder_move:
            print(f"bestmove {best_move} ponder {ponder_move}")
        elif best_move:
            print(f"bestmove {best_move}")
        else:
            print(f"bestmove {legal_moves[0].uci()}")


def uci_loop(model_path="guitarfish_v2.gm", book_path="book.bin"):
    evaluator = Evaluator(model_path)
    searcher = Searcher(evaluator, book_path)
    board = chess.Board()

    author = evaluator.metadata.get("author", "MemeViber")
    desc = evaluator.metadata.get("description", "INT8")

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
