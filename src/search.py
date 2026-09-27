import json
import math
import os
import time
import numpy as np

import src.bitboard as chess

PIECE_VALS = np.array([0, 100, 300, 300, 500, 900, 20000], dtype=np.int32)

MATE_VALUE = 30000
MAX_PLY = 64

DEFAULT_PARAMS = {
    "rfp_base": 80.0,
    "rfp_improving": 20.0,
    "nmp_base": 3.0,
    "nmp_divisor": 4.0,
    "lmr_base": 0.75,
    "lmr_divisor": 2.25,
    "aspiration_delta": 35.0,
    "se_margin_mult": 2.0,
}


def build_lmr_table(base, divisor):
    table = np.zeros((64, 64), dtype=np.int32)
    for d in range(1, 64):
        for mc in range(1, 64):
            table[d, mc] = int(base + math.log(d) * math.log(mc) / divisor)
    return table


class FastHistory:
    def __init__(self):
        self.table = np.zeros((2, 64, 64), dtype=np.int32)

    def clear(self):
        self.table.fill(0)

    def get(self, key, default=0):
        t = 1 if key[0] else 0
        return int(self.table[t, key[1], key[2]])

    def add(self, turn, from_sq, to_sq, bonus):
        t = 1 if turn else 0
        self.table[t, from_sq, to_sq] += bonus

    def decay(self):
        self.table[:] = (self.table * 3) // 4


class FastCounterMoves:
    def __init__(self):
        self.table = np.zeros(4096, dtype=np.int32)

    def clear(self):
        self.table.fill(0)

    def get_move(self, prev_move):
        if prev_move is None:
            return None
        idx = (prev_move.from_square) | (prev_move.to_square << 6)
        val = self.table[idx]
        return val if val != 0 else None

    def store(self, prev_move, move):
        if prev_move is not None:
            idx = (prev_move.from_square) | (prev_move.to_square << 6)
            self.table[idx] = move._value


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
            stored_move = move if move is not None else (entry[4] if entry else None)
            self.table[idx] = (zobrist, depth, flag, score, stored_move)

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
    def __init__(
        self,
        evaluator,
        book_path="book.bin",
        params=None,
        params_file="search_params.json",
        tt_size_mb=256,
    ):
        self.evaluator = evaluator
        self.book = chess.polyglot.open_reader(book_path)
        self.tt = TranspositionTable(size_mb=tt_size_mb)

        self.params = DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)
        elif os.path.exists(params_file):
            try:
                with open(params_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.params.update(loaded)
            except Exception:
                pass

        self.lmr_table = build_lmr_table(
            float(self.params["lmr_base"]), float(self.params["lmr_divisor"])
        )

        self.history = FastHistory()
        self.counter_moves = FastCounterMoves()
        self.killers = np.zeros((MAX_PLY, 2), dtype=np.int64)

        self.eval_stack = [0] * MAX_PLY
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.stop = False
        self.sel_depth = 0
        self.expected_opponent_move = None
        self.premove_reply = None
        self.pv_lines = [[] for _ in range(MAX_PLY)]

    def _build_mailbox(self, pieces):
        mailbox = [0] * 64
        for idx in range(12):
            bb = int(pieces[idx])
            p_val = (idx % 6) + 1
            while bb:
                sq = (bb & -bb).bit_length() - 1
                mailbox[sq] = p_val
                bb &= bb - 1
        return mailbox

    def score_moves_fast(self, board, moves, depth, tt_move_val, prev_move, mailbox):
        t_idx = 1 if board.turn else 0
        hist_t = self.history.table[t_idx]
        k0 = self.killers[depth, 0]
        k1 = self.killers[depth, 1]
        cm_val = self.counter_moves.get_move(prev_move)
        ep_sq = board.ep_square
        pieces = board._pieces

        scored = []
        for move in moves:
            m_val = move._value
            if m_val == tt_move_val:
                scored.append((2_000_000, move))
                continue

            from_sq = move.from_square
            to_sq = move.to_square
            victim = mailbox[to_sq]

            if victim != 0 or to_sq == ep_sq:
                attacker = mailbox[from_sq] or 1
                v_type = victim if victim != 0 else 1
                mvv_lva = PIECE_VALS[v_type] * 10 - PIECE_VALS[attacker]

                if chess.fast_see(pieces, board.turn, from_sq, to_sq, 0):
                    scored.append((1_000_000 + mvv_lva, move))
                else:
                    scored.append((-100_000 + mvv_lva, move))
            elif m_val == k0:
                scored.append((900_000, move))
            elif m_val == k1:
                scored.append((800_000, move))
            elif cm_val is not None and m_val == cm_val:
                scored.append((750_000, move))
            else:
                scored.append((int(hist_t[from_sq, to_sq]), move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored]

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

        captures = board.generate_legal_captures()
        if not captures:
            return alpha

        mailbox = self._build_mailbox(board._pieces)
        pieces = board._pieces

        scored_captures = []
        for m in captures:
            see_val = chess.see(pieces, board.turn, m.from_square, m.to_square)
            if see_val < 0:
                continue

            victim = mailbox[m.to_square] or 1
            attacker = mailbox[m.from_square] or 1
            mvv_lva = PIECE_VALS[victim] * 10 - PIECE_VALS[attacker]
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

    def negamax(
        self,
        board,
        depth,
        alpha,
        beta,
        ply,
        allow_nmp=True,
        prev_move=None,
        excluded_move=None,
    ):
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
        tt_entry = self.tt.get(zobrist)
        tt_move = None
        tt_move_val = 0
        tt_score = None
        tt_depth = 0
        tt_flag = -1

        if tt_entry is not None:
            _, tt_depth, tt_flag, tt_score, tt_move = tt_entry
            if tt_move is not None:
                tt_move_val = tt_move._value

            if tt_score > MATE_VALUE - 1000:
                tt_score -= ply
            elif tt_score < -MATE_VALUE + 1000:
                tt_score += ply

            if excluded_move is None and tt_depth >= depth:
                if tt_flag == 0:
                    if ply > 0:
                        return tt_score
                elif tt_flag == 1 and tt_score <= alpha:
                    if ply > 0:
                        return alpha
                elif tt_flag == 2 and tt_score >= beta:
                    if ply > 0:
                        return beta

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

        if (
            not in_check
            and depth <= 6
            and abs(beta) < MATE_VALUE - 1000
            and excluded_move is None
        ):
            rfp_margin = (
                self.params["rfp_base"] - self.params["rfp_improving"] * int(improving)
            ) * depth
            if static_eval - rfp_margin >= beta:
                return static_eval - rfp_margin

        if (
            allow_nmp
            and not in_check
            and ply > 0
            and depth >= 3
            and static_eval >= beta
            and excluded_move is None
        ):
            if board.piece_count() > 5:
                null_move = chess.Move.null()
                board.push(null_move)
                r_div = max(1.0, float(self.params["nmp_divisor"]))
                R = int(
                    self.params["nmp_base"] + (depth // r_div) + (1 if improving else 0)
                )
                null_score = -self.negamax(
                    board,
                    depth - 1 - R,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    False,
                    None,
                    None,
                )
                board.pop()

                if self.stop:
                    return 0
                if null_score >= beta:
                    return beta

        singular_extension = False
        if (
            depth >= 8
            and tt_move is not None
            and excluded_move is None
            and ply > 0
            and tt_depth >= depth - 3
            and tt_flag != 1
            and abs(tt_score) < MATE_VALUE - 1000
        ):
            singular_margin = self.params["se_margin_mult"] * depth
            singular_beta = tt_score - singular_margin
            singular_depth = (depth - 1) // 2

            s_score = self.negamax(
                board,
                singular_depth,
                singular_beta - 1,
                singular_beta,
                ply,
                False,
                prev_move,
                excluded_move=tt_move,
            )

            if s_score < singular_beta:
                singular_extension = True

        legal_moves = board.legal_moves
        if not legal_moves:
            if excluded_move:
                return alpha
            return -MATE_VALUE + ply if in_check else 0

        mailbox = self._build_mailbox(board._pieces)
        sorted_moves = self.score_moves_fast(
            board, legal_moves, depth, tt_move_val, prev_move, mailbox
        )

        best_score = -MATE_VALUE
        best_move_found = None
        moves_searched = 0
        alpha_orig = alpha
        quiets_searched = []

        for move in sorted_moves:
            if excluded_move is not None and move == excluded_move:
                continue

            is_cap = (mailbox[move.to_square] != 0) or (
                move.to_square == board.ep_square
            )
            board.push(move)
            moves_searched += 1

            extension = 0
            if board.is_check() and ply < 32:
                extension = 1
            elif tt_move is not None and move == tt_move and singular_extension:
                extension = 1

            new_depth = depth - 1 + extension

            if moves_searched == 1:
                score = -self.negamax(
                    board, new_depth, -beta, -alpha, ply + 1, True, move, None
                )
            else:
                r = 0
                if depth >= 3 and moves_searched > 1 and not is_cap:
                    d_idx = min(depth, 63)
                    m_idx = min(moves_searched, 63)
                    r = int(self.lmr_table[d_idx, m_idx])
                    if improving:
                        r -= 1
                    t_idx = 1 if board.turn else 0
                    if (
                        self.history.table[t_idx, move.from_square, move.to_square]
                        > 5000
                    ):
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
                    None,
                )
                if score > alpha and (score < beta or r > 0):
                    score = -self.negamax(
                        board, new_depth, -beta, -alpha, ply + 1, True, move, None
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
                    m_val = move._value
                    if m_val != self.killers[depth, 0]:
                        self.killers[depth, 1] = self.killers[depth, 0]
                        self.killers[depth, 0] = m_val

                    bonus = depth * depth
                    self.history.add(
                        board.turn, move.from_square, move.to_square, bonus
                    )

                    if prev_move:
                        self.counter_moves.store(prev_move, move)

                    for qm in quiets_searched:
                        if qm != move:
                            self.history.add(
                                board.turn, qm.from_square, qm.to_square, -bonus
                            )
                break

        if excluded_move is None:
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

    def search(self, board, time_limit=5.0, fixed_depth=None, silent=False):
        if not fixed_depth or fixed_depth > 1:
            book_move = self.book.probe(board)
            if book_move:
                if not silent:
                    print(f"info depth 1 score cp 20 time 1 pv {book_move.uci()}")
                    print(f"bestmove {book_move.uci()}")
                return book_move.uci()

        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.stop = False
        self.sel_depth = 0
        self.killers.fill(0)
        self.history.decay()

        legal_moves = board.legal_moves
        if not legal_moves:
            if not silent:
                print("bestmove (none)")
            return "(none)"

        best_move = legal_moves[0].uci()
        ponder_move = None
        best_val_prev = None
        max_depth = fixed_depth if fixed_depth else 64

        for depth in range(1, max_depth + 1):
            self.sel_depth = 0

            if depth >= 5 and best_val_prev is not None:
                delta = int(self.params["aspiration_delta"])
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
                ponder_move = pv_list[1].uci() if len(pv_list) > 1 else None

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

            if not silent:
                print(
                    f"info depth {depth} seldepth {self.sel_depth} score {score_str} "
                    f"nodes {self.nodes} nps {nps} time {int(elapsed * 1000)} pv {pv_str}"
                )

            if not fixed_depth and elapsed > self.time_limit * 0.6:
                break

        if not silent:
            if best_move and ponder_move:
                print(f"bestmove {best_move} ponder {ponder_move}")
            elif best_move:
                print(f"bestmove {best_move}")
            else:
                print(f"bestmove {legal_moves[0].uci()}")

        return best_move
