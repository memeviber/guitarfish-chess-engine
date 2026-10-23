import gc
import math
import time

import numpy as np

import src.bitboard as chess
from src.tablebase import OfflineTablebase, OnlineTablebase

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
    "razor_margin": 342.0,
    "probcut_margin": 200.0,
    "see_prune_margin": 80.0,
    "lmp_base": 4.0,
    "lmp_depth_factor": 1.0,
    "child_futility_margin": 90.0,
    "multicut_depth": 6.0,
    "multicut_m": 8.0,
    "multicut_c": 3.0,
    "multicut_r": 3.0,
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

        if entry is None:
            self.table[idx] = [zobrist, depth, flag, score, move]
        elif depth >= entry[1] or flag == 0:
            stored_move = move if move is not None else entry[4]
            entry[0] = zobrist
            entry[1] = depth
            entry[2] = flag
            entry[3] = score
            entry[4] = stored_move

    def probe(self, zobrist, depth, alpha, beta, ply):
        entry = self.table[zobrist & self.mask]
        if entry is not None and entry[0] == zobrist:
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
        if entry is not None and entry[0] == zobrist:
            return entry
        return None

    def clear(self):
        self.table = [None] * self.size


class Searcher:
    def __init__(
        self,
        evaluator,
        book_path="book.bin",
        tt_size_mb=256,
    ):
        self.evaluator = evaluator
        self.book = chess.polyglot.open_reader(book_path)
        self.tt = TranspositionTable(size_mb=tt_size_mb)
        self.tt_size_mb = tt_size_mb

        self.params = DEFAULT_PARAMS.copy()

        self.lmr_table = build_lmr_table(
            float(self.params["lmr_base"]), float(self.params["lmr_divisor"])
        )

        self.history = FastHistory()
        self.counter_moves = FastCounterMoves()
        self.killers = np.zeros((MAX_PLY * 2, 2), dtype=np.int64)

        self.eval_stack = [0] * MAX_PLY
        self.nodes = 0
        self.start_time = 0.0
        self.hard_time_limit = 0.0
        self.soft_time_limit = 0.0
        self.stop = False
        self.sel_depth = 0
        self.expected_opponent_move = None
        self.premove_reply = None

        self.pv_table = [[None] * MAX_PLY for _ in range(MAX_PLY)]
        self.pv_length = [0] * MAX_PLY

        self._mailbox = [[0] * 64 for _ in range(MAX_PLY * 2)]
        self._mailbox_pieces = np.zeros((MAX_PLY * 2, 12), dtype=np.uint64)

        self.tablebase = OnlineTablebase(timeout=1.2)
        self.syzygy_tablebase = None
        self.online_tablebase_enabled = True
        self.own_book = True

    def set_hash_size(self, size_mb):
        self.tt = TranspositionTable(size_mb=size_mb)
        self.tt_size_mb = size_mb

    def set_syzygy_path(self, path):
        tablebase = OfflineTablebase(path) if path else None
        previous = self.syzygy_tablebase
        self.syzygy_tablebase = tablebase
        if previous is not None:
            previous.close()

    def close(self):
        if self.syzygy_tablebase is not None:
            self.syzygy_tablebase.close()
            self.syzygy_tablebase = None

    def _update_mailbox(self, pieces, ply):
        slot = min(ply, MAX_PLY * 2 - 1)
        mb = self._mailbox[slot]
        previous = self._mailbox_pieces[slot]

        for idx in range(12):
            old_bb = int(previous[idx])
            bb = int(pieces[idx])
            removed = old_bb & ~bb
            while removed:
                sq = (removed & -removed).bit_length() - 1
                mb[sq] = 0
                removed &= removed - 1

        for idx in range(12):
            old_bb = int(previous[idx])
            bb = int(pieces[idx])
            p_val = (idx % 6) + 1
            added = bb & ~old_bb
            while added:
                sq = (added & -added).bit_length() - 1
                mb[sq] = p_val
                added &= added - 1
            previous[idx] = bb
        return mb

    def score_moves_fast(self, board, moves, depth, tt_move_val, prev_move, mailbox):
        t_idx = 1 if board.turn else 0
        hist_t = self.history.table[t_idx]
        k0 = self.killers[depth, 0] if depth < self.killers.shape[0] else 0
        k1 = self.killers[depth, 1] if depth < self.killers.shape[0] else 0
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
        return scored

    def quiescence(self, board, alpha, beta, ply=0):
        self.nodes += 1
        if (self.nodes & 1023) == 0 and (
            time.perf_counter() - self.start_time
        ) >= self.hard_time_limit:
            self.stop = True
        if self.stop:
            return 0

        if ply > self.sel_depth:
            self.sel_depth = ply

        in_check = board.is_check()
        if in_check:
            moves = board.legal_moves
            if not moves:
                return -MATE_VALUE + ply
            if ply >= MAX_PLY - 1:
                best_score = -MATE_VALUE
                for move in moves:
                    board.push(move)
                    child_moves = board.legal_moves
                    if not child_moves:
                        score = MATE_VALUE - (ply + 1) if board.is_check() else 0
                    else:
                        score = -self.evaluator.evaluate(board)
                    board.pop(return_move=False)
                    best_score = max(best_score, score)
                return best_score
        else:
            stand_pat = self.evaluator.evaluate(board)
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat

            if ply >= MAX_PLY - 1 or ply >= 32:
                return stand_pat

            moves, has_legal_move = board.generate_legal_noisy_moves()
            if not moves:
                return 0 if not has_legal_move else alpha

        if ply >= 32:
            best_score = -MATE_VALUE
            for move in moves:
                board.push(move)
                child_moves = board.legal_moves
                if not child_moves:
                    score = MATE_VALUE - (ply + 1) if board.is_check() else 0
                else:
                    score = -self.evaluator.evaluate(board)
                board.pop(return_move=False)
                best_score = max(best_score, score)
            return best_score

        mailbox = self._update_mailbox(board._pieces, ply)
        pieces = board._pieces

        scored_captures = []
        for move in moves:
            is_capture = mailbox[move.to_square] != 0 or (
                move.to_square == board.ep_square and mailbox[move.from_square] == 1
            )
            victim = mailbox[move.to_square] or (1 if is_capture else 0)
            attacker = mailbox[move.from_square] or 1
            mvv_lva = PIECE_VALS[victim] * 10 - PIECE_VALS[attacker]
            promotion_gain = (
                PIECE_VALS[move.promotion] - PIECE_VALS[1]
                if move.promotion is not None
                else 0
            )
            capture_gain = (PIECE_VALS[victim] if is_capture else 0) + promotion_gain
            scored_captures.append(
                (mvv_lva + promotion_gain, move, is_capture, int(capture_gain))
            )

        if not scored_captures:
            return alpha

        if (
            not in_check
            and stand_pat + max(item[3] for item in scored_captures) + 100 < alpha
        ):
            return alpha

        scored_captures.sort(key=lambda x: x[0], reverse=True)

        for _, move, is_capture, _ in scored_captures:
            if not in_check and is_capture:
                see_val = chess.see(
                    pieces, board.turn, move.from_square, move.to_square
                )
                if see_val < 0:
                    continue

            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop(return_move=False)

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
            self.pv_length[ply] = ply

        if ply > 0 and (board.is_repetition(2) or board.can_claim_fifty_moves()):
            return 0

        if (self.nodes & 1023) == 0 and (
            time.perf_counter() - self.start_time
        ) >= self.hard_time_limit:
            self.stop = True
        if self.stop:
            return 0

        if ply >= MAX_PLY - 1:
            return self.evaluator.evaluate(board)

        zobrist = int(board._hash)
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
            ply > 0
            and not in_check
            and depth <= 3
            and excluded_move is None
            and static_eval + self.params["razor_margin"] * depth <= alpha
        ):
            return self.quiescence(board, alpha, beta, ply)

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
                board.pop(return_move=False)

                if self.stop:
                    return 0
                if null_score >= beta:
                    return beta

        if (
            allow_nmp
            and ply > 0
            and not in_check
            and depth >= int(self.params["multicut_depth"])
            and abs(beta) < MATE_VALUE - 1000
            and excluded_move is None
        ):
            mc_r = int(self.params["multicut_r"])
            mc_depth = depth - 1 - mc_r
            if mc_depth > 0:
                mc_mailbox = self._update_mailbox(board._pieces, ply)
                mc_legal_moves = board.legal_moves
                if mc_legal_moves:
                    mc_sorted = self.score_moves_fast(
                        board, mc_legal_moves, depth, tt_move_val, prev_move, mc_mailbox
                    )
                    mc_moves_to_test = int(self.params["multicut_m"])
                    mc_threshold = int(self.params["multicut_c"])
                    mc_cuts = 0

                    for _, mc_move in mc_sorted[:mc_moves_to_test]:
                        board.push(mc_move)
                        mc_score = -self.negamax(
                            board,
                            mc_depth,
                            -beta,
                            -beta + 1,
                            ply + 1,
                            False,
                            mc_move,
                            None,
                        )
                        board.pop(return_move=False)

                        if self.stop:
                            return 0

                        if mc_score >= beta:
                            mc_cuts += 1
                            if mc_cuts >= mc_threshold:
                                return beta

        if (
            ply > 0
            and not in_check
            and depth >= 4
            and beta < MATE_VALUE - 500
            and excluded_move is None
        ):
            prob_margin = int(round(self.params["probcut_margin"]))
            prob_beta = beta + prob_margin
            for move in board.generate_legal_captures()[:8]:
                board.push(move)
                qscore = -self.quiescence(board, -prob_beta, -prob_beta + 1, ply + 1)
                score = -MATE_VALUE
                if not self.stop and qscore >= prob_beta:
                    score = -self.negamax(
                        board,
                        depth - 3,
                        -prob_beta,
                        -prob_beta + 1,
                        ply + 1,
                        True,
                        move,
                        None,
                    )
                board.pop(return_move=False)
                if self.stop:
                    return 0
                if score >= prob_beta:
                    return score - prob_margin

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

        mailbox = self._update_mailbox(board._pieces, ply)
        sorted_moves = self.score_moves_fast(
            board, legal_moves, depth, tt_move_val, prev_move, mailbox
        )

        cm_val = self.counter_moves.get_move(prev_move)
        best_score = -MATE_VALUE
        best_move_found = None
        moves_searched = 0
        alpha_orig = alpha
        quiets_searched = []

        for _, move in sorted_moves:
            if excluded_move is not None and move == excluded_move:
                continue

            m_val = move._value
            is_cap = (mailbox[move.to_square] != 0) or (
                move.to_square == board.ep_square
            )
            see_pruning_candidate = (
                excluded_move is None
                and is_cap
                and not in_check
                and move.promotion is None
                and beta == alpha + 1
                and depth <= 3
                and moves_searched > 0
                and not (tt_move == move and singular_extension)
            )
            see_passes = not see_pruning_candidate or chess.fast_see(
                board._pieces,
                board.turn,
                move.from_square,
                move.to_square,
                -int(round(self.params["see_prune_margin"] * depth)),
            )
            board.push(move)
            moves_searched += 1

            gives_check = board.is_check()
            quiet = (
                not is_cap
                and not gives_check
                and not in_check
                and move.promotion is None
            )

            if (
                excluded_move is None
                and is_cap
                and not in_check
                and not gives_check
                and move.promotion is None
                and beta == alpha + 1
                and depth <= 3
                and moves_searched > 1
                and not (tt_move == move and singular_extension)
                and not see_passes
            ):
                board.pop(return_move=False)
                continue

            lmp_limit = int(
                round(
                    self.params["lmp_base"]
                    + self.params["lmp_depth_factor"] * depth * depth
                )
            )
            if (
                excluded_move is None
                and quiet
                and beta == alpha + 1
                and depth <= 3
                and moves_searched - 1 >= lmp_limit
            ):
                board.pop(return_move=False)
                continue

            if (
                excluded_move is None
                and quiet
                and moves_searched > 1
                and depth <= 3
                and static_eval + self.params["child_futility_margin"] * depth <= alpha
            ):
                board.pop(return_move=False)
                continue

            extension = 0
            if gives_check and ply < 32:
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
                if depth >= 3 and moves_searched > 1:
                    d_idx = min(depth, 63)
                    m_idx = min(moves_searched, 63)

                    if not is_cap:
                        r = int(self.lmr_table[d_idx, m_idx])

                        if gives_check:
                            r -= 1
                        if move.promotion is not None:
                            r -= 1
                        if depth < self.killers.shape[0]:
                            if (
                                m_val == self.killers[depth, 0]
                                or m_val == self.killers[depth, 1]
                            ):
                                r -= 1
                        if cm_val is not None and m_val == cm_val:
                            r -= 1

                        t_idx = 1 if board.turn else 0
                        hist_val = int(
                            self.history.table[t_idx, move.from_square, move.to_square]
                        )
                        if hist_val > 8000:
                            r -= 2
                        elif hist_val > 2000:
                            r -= 1
                        elif hist_val < -4000:
                            r += 1
                        elif hist_val < -10000:
                            r += 2

                        if improving:
                            r -= 1
                        else:
                            r += 1

                        if (
                            not in_check
                            and static_eval - self.params["rfp_base"] * depth >= beta
                        ):
                            r += 1
                    else:
                        if depth >= 5 and moves_searched > 6 and not gives_check:
                            r = 1

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

            board.pop(return_move=False)

            if self.stop:
                return 0

            if not is_cap:
                quiets_searched.append(move)

            if score > best_score:
                best_score = score
                best_move_found = move
                if ply < MAX_PLY - 1:
                    self.pv_table[ply][ply] = move
                    nxt_len = self.pv_length[ply + 1]
                    for j in range(ply + 1, nxt_len):
                        self.pv_table[ply][j] = self.pv_table[ply + 1][j]
                    self.pv_length[ply] = max(ply + 1, nxt_len)

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not is_cap and depth < self.killers.shape[0]:
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
            zobrist = int(curr._hash)
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

    def _get_fen_from_board(self, board):
        rank_strs = []
        syms = {
            1: ("P", "p"),
            2: ("N", "n"),
            3: ("B", "b"),
            4: ("R", "r"),
            5: ("Q", "q"),
            6: ("K", "k"),
        }
        for r in range(7, -1, -1):
            empty, r_str = 0, ""
            for f in range(8):
                sq = r * 8 + f
                pt = board.piece_at(sq)
                if pt is None:
                    empty += 1
                else:
                    if empty > 0:
                        r_str += str(empty)
                        empty = 0
                    r_str += syms[pt.piece_type][0 if pt.color else 1]
            if empty > 0:
                r_str += str(empty)
            rank_strs.append(r_str)
        turn_str = "w" if board.turn == chess.WHITE else "b"
        c_str = ""
        if board.castling_rights & 1:
            c_str += "K"
        if board.castling_rights & 2:
            c_str += "Q"
        if board.castling_rights & 4:
            c_str += "k"
        if board.castling_rights & 8:
            c_str += "q"
        if not c_str:
            c_str = "-"
        ep_str = "-"
        if board.ep_square >= 0:
            ep_str = chr(97 + (board.ep_square & 7)) + str(1 + board.ep_square // 8)
        return f"{'/'.join(rank_strs)} {turn_str} {c_str} {ep_str} {board.halfmove_clock} {board.fullmove_number}"

    def search(
        self, board, time_limit=5.0, fixed_depth=None, silent=False, shut_up=False
    ):
        if self.own_book and (not fixed_depth or fixed_depth > 1 and not shut_up):
            book_move = self.book.probe(board)
            if book_move:
                if not silent:
                    print(f"info depth 1 score cp 20 time 1 pv {book_move.uci()}")
                    print(f"bestmove {book_move.uci()}")

                return book_move.uci()

        self.nodes = 0
        self.start_time = time.perf_counter()

        raw_limit = time_limit if time_limit is not None else 1e9
        safety_buffer = min(0.15, raw_limit * 0.08) if raw_limit < 60.0 else 0.0
        self.hard_time_limit = max(0.05, raw_limit - safety_buffer)
        self.soft_time_limit = self.hard_time_limit * 0.45

        self.stop = False
        self.sel_depth = 0
        self.killers.fill(0)
        self.history.decay()

        legal_moves = board.legal_moves
        if not legal_moves:
            if shut_up:
                return
            if not silent:
                print("bestmove (none)")
            return "(none)"

        if board.piece_count() <= 7 and self.hard_time_limit > 1.5:
            fen_str = self._get_fen_from_board(board)
            tb = None
            if self.syzygy_tablebase is not None:
                tb = self.syzygy_tablebase.probe(fen_str)
            if tb is None and self.online_tablebase_enabled:
                tb = self.tablebase.probe(fen_str)
            if tb and tb["best_move"]:
                best_m = tb["best_move"]
                score_str = f"cp {tb['score_cp']}"
                cat = tb["category"].upper()
                dt_info = f"DTM:{tb['dtm']}" if tb["dtm"] else f"DTZ:{tb['dtz']}"

                if shut_up:
                    return

                if not silent:
                    print(
                        f"info depth 100 score {score_str} pv {best_m} ({cat} in {dt_info})"
                    )
                    print(f"bestmove {best_m}")

                return best_m

        best_move = legal_moves[0].uci()
        ponder_move = None
        best_val_prev = None
        max_depth = fixed_depth if fixed_depth else 64

        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()

        try:
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
                            score = self.negamax(
                                board, depth, -MATE_VALUE, MATE_VALUE, 0
                            )
                            break
                else:
                    score = self.negamax(board, depth, -MATE_VALUE, MATE_VALUE, 0)

                if self.stop:
                    break

                best_val_prev = score

                pv_list = self.get_pv_from_tt(board)
                if not pv_list and self.pv_length[0] > 0:
                    pv_list = [self.pv_table[0][i] for i in range(self.pv_length[0])]

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

                elapsed = max(time.perf_counter() - self.start_time, 1e-6)
                nps = int(self.nodes / elapsed)
                score_str = (
                    f"mate {(MATE_VALUE - abs(score) + 1) // 2 if score > 0 else -((MATE_VALUE - abs(score) + 1) // 2)}"
                    if abs(score) > MATE_VALUE - 1000
                    else f"cp {score}"
                )

                if shut_up:
                    continue

                if not silent:
                    print(
                        f"info depth {depth} seldepth {self.sel_depth} score {score_str} "
                        f"nodes {self.nodes} nps {nps} time {int(elapsed * 1000)} pv {pv_str}"
                    )

                if (
                    not fixed_depth
                    and self.hard_time_limit < 1e8
                    and elapsed >= self.soft_time_limit
                ):
                    break
        finally:
            if gc_was_enabled:
                gc.enable()

        if shut_up:
            return

        if not silent:
            if best_move and ponder_move:
                print(f"bestmove {best_move} ponder {ponder_move}")
            elif best_move:
                print(f"bestmove {best_move}")
            else:
                print(f"bestmove {legal_moves[0].uci()}")

        return best_move
