import io
import json
import os
import zipfile

import numpy as np
from numba import njit

import src.bitboard as chess

DEFAULT_WDL_SCALE = 410.0
MAX_PLY = 96

SQUARE_TO_BUCKET = np.zeros(64, dtype=np.int64)
for sq in range(64):
    f = sq % 8
    r = sq // 8
    f_sym = f if f < 4 else 7 - f
    f_group = 0 if f_sym < 2 else 1
    r_group = r // 2
    SQUARE_TO_BUCKET[sq] = r_group * 2 + f_group


@njit(cache=True, fastmath=True)
def _refresh_acc_white(pieces, ft_w, ft_b, out_acc, sq_to_bucket):
    k_bits = pieces[5]
    wk_sq = chess._lsb_index(k_bits)
    bucket = sq_to_bucket[wk_sq]
    offset = bucket * 768

    l1_size = ft_b.shape[0]
    for j in range(l1_size):
        out_acc[j] = ft_b[j]

    for pt in range(6):
        bb = pieces[pt]
        feat_base = offset + pt * 64
        while bb:
            bit = bb & (np.uint64(0) - bb)
            sq = chess._lsb_index(bit)
            idx = feat_base + sq
            for j in range(l1_size):
                out_acc[j] += ft_w[idx, j]
            bb ^= bit

    for pt in range(6):
        bb = pieces[pt + 6]
        feat_base = offset + (pt + 6) * 64
        while bb:
            bit = bb & (np.uint64(0) - bb)
            sq = chess._lsb_index(bit)
            idx = feat_base + sq
            for j in range(l1_size):
                out_acc[j] += ft_w[idx, j]
            bb ^= bit

    return bucket


@njit(cache=True, fastmath=True)
def _refresh_acc_black(pieces, ft_w, ft_b, out_acc, sq_to_bucket):
    k_bits = pieces[11]
    bk_sq = chess._lsb_index(k_bits)
    bucket = sq_to_bucket[bk_sq ^ 56]
    offset = bucket * 768

    l1_size = ft_b.shape[0]
    for j in range(l1_size):
        out_acc[j] = ft_b[j]

    for pt in range(6):
        bb = pieces[pt + 6]
        feat_base = offset + pt * 64
        while bb:
            bit = bb & (np.uint64(0) - bb)
            sq = chess._lsb_index(bit)
            idx = feat_base + (sq ^ 56)
            for j in range(l1_size):
                out_acc[j] += ft_w[idx, j]
            bb ^= bit

    for pt in range(6):
        bb = pieces[pt]
        feat_base = offset + (pt + 6) * 64
        while bb:
            bit = bb & (np.uint64(0) - bb)
            sq = chess._lsb_index(bit)
            idx = feat_base + (sq ^ 56)
            for j in range(l1_size):
                out_acc[j] += ft_w[idx, j]
            bb ^= bit

    return bucket


@njit(cache=True, fastmath=True)
def _update_incremental(
    parent_pieces,
    curr_pieces,
    parent_w_acc,
    parent_b_acc,
    parent_w_bkt,
    parent_b_bkt,
    out_w_acc,
    out_b_acc,
    m_val,
    moving_idx,
    captured_idx,
    ft_w,
    ft_b,
    sq_to_bucket,
):
    l1_size = ft_b.shape[0]
    from_sq = m_val & 63
    to_sq = (m_val >> 6) & 63
    promo = (m_val >> 12) & 7

    new_w_bkt = parent_w_bkt
    if moving_idx == 5:
        target_bkt = sq_to_bucket[to_sq]
        if target_bkt != parent_w_bkt:
            new_w_bkt = _refresh_acc_white(
                curr_pieces, ft_w, ft_b, out_w_acc, sq_to_bucket
            )
        else:
            for j in range(l1_size):
                out_w_acc[j] = parent_w_acc[j]
            off = parent_w_bkt * 768
            old_idx = off + 5 * 64 + from_sq
            new_idx = off + 5 * 64 + to_sq
            for j in range(l1_size):
                out_w_acc[j] += ft_w[new_idx, j] - ft_w[old_idx, j]
            if abs(to_sq - from_sq) == 2:
                r_from, r_to = (7, 5) if to_sq > from_sq else (0, 3)
                r_old = off + 3 * 64 + r_from
                r_new = off + 3 * 64 + r_to
                for j in range(l1_size):
                    out_w_acc[j] += ft_w[r_new, j] - ft_w[r_old, j]
    else:
        for j in range(l1_size):
            out_w_acc[j] = parent_w_acc[j]
        off = parent_w_bkt * 768
        old_idx = off + moving_idx * 64 + from_sq
        placed_idx = (promo - 1) if promo != 0 else moving_idx
        new_idx = off + placed_idx * 64 + to_sq
        for j in range(l1_size):
            out_w_acc[j] += ft_w[new_idx, j] - ft_w[old_idx, j]

        if captured_idx >= 0:
            if (
                moving_idx == 0
                and captured_idx == 6
                and to_sq != from_sq
                and (to_sq & 7) != (from_sq & 7)
            ):
                cap_sq = to_sq - 8
                cap_feat = off + 6 * 64 + cap_sq
            else:
                cap_feat = off + captured_idx * 64 + to_sq
            for j in range(l1_size):
                out_w_acc[j] -= ft_w[cap_feat, j]

        if moving_idx == 11 and abs(to_sq - from_sq) == 2:
            r_from, r_to = (63, 61) if to_sq > from_sq else (56, 59)
            r_old = off + 9 * 64 + r_from
            r_new = off + 9 * 64 + r_to
            for j in range(l1_size):
                out_w_acc[j] += ft_w[r_new, j] - ft_w[r_old, j]

    new_b_bkt = parent_b_bkt
    if moving_idx == 11:
        target_bkt = sq_to_bucket[to_sq ^ 56]
        if target_bkt != parent_b_bkt:
            new_b_bkt = _refresh_acc_black(
                curr_pieces, ft_w, ft_b, out_b_acc, sq_to_bucket
            )
        else:
            for j in range(l1_size):
                out_b_acc[j] = parent_b_acc[j]
            off = parent_b_bkt * 768
            old_idx = off + 5 * 64 + (from_sq ^ 56)
            new_idx = off + 5 * 64 + (to_sq ^ 56)
            for j in range(l1_size):
                out_b_acc[j] += ft_w[new_idx, j] - ft_w[old_idx, j]
            if abs(to_sq - from_sq) == 2:
                r_from, r_to = (63, 61) if to_sq > from_sq else (56, 59)
                r_old = off + 3 * 64 + (r_from ^ 56)
                r_new = off + 3 * 64 + (r_to ^ 56)
                for j in range(l1_size):
                    out_b_acc[j] += ft_w[r_new, j] - ft_w[r_old, j]
    else:
        for j in range(l1_size):
            out_b_acc[j] = parent_b_acc[j]
        off = parent_b_bkt * 768
        b_moving_idx = (moving_idx - 6) if moving_idx >= 6 else (moving_idx + 6)
        old_idx = off + b_moving_idx * 64 + (from_sq ^ 56)
        placed_idx = (
            (promo - 1)
            if (promo != 0 and moving_idx >= 6)
            else ((promo + 5) if promo != 0 else b_moving_idx)
        )
        new_idx = off + placed_idx * 64 + (to_sq ^ 56)
        for j in range(l1_size):
            out_b_acc[j] += ft_w[new_idx, j] - ft_w[old_idx, j]

        if captured_idx >= 0:
            b_cap_idx = (captured_idx - 6) if captured_idx >= 6 else (captured_idx + 6)
            if (
                moving_idx == 6
                and captured_idx == 0
                and to_sq != from_sq
                and (to_sq & 7) != (from_sq & 7)
            ):
                cap_sq = to_sq + 8
                cap_feat = off + b_cap_idx * 64 + (cap_sq ^ 56)
            else:
                cap_feat = off + b_cap_idx * 64 + (to_sq ^ 56)
            for j in range(l1_size):
                out_b_acc[j] -= ft_w[cap_feat, j]

        if moving_idx == 5 and abs(to_sq - from_sq) == 2:
            r_from, r_to = (7, 5) if to_sq > from_sq else (0, 3)
            r_old = off + 9 * 64 + (r_from ^ 56)
            r_new = off + 9 * 64 + (r_to ^ 56)
            for j in range(l1_size):
                out_b_acc[j] += ft_w[r_new, j] - ft_w[r_old, j]

    return new_w_bkt, new_b_bkt


@njit(cache=True, fastmath=True)
def _forward(acc_stm, acc_opp, w_fc1, b_fc1, w_fc2, b_fc2, w_out, b_out):
    l1_half = acc_stm.shape[0] // 2
    pooled = np.empty(acc_stm.shape[0], dtype=np.float32)

    for i in range(l1_half):
        p1 = min(max(acc_stm[i], 0.0), 1.0)
        p2 = min(max(acc_stm[i + l1_half], 0.0), 1.0)
        pooled[i] = p1 * p2
        p3 = min(max(acc_opp[i], 0.0), 1.0)
        p4 = min(max(acc_opp[i + l1_half], 0.0), 1.0)
        pooled[i + l1_half] = p3 * p4

    z1 = np.empty(64, dtype=np.float32)
    for i in range(64):
        s = b_fc1[i]
        for j in range(acc_stm.shape[0]):
            s += w_fc1[i, j] * pooled[j]
        z1[i] = s

    h1 = np.empty(128, dtype=np.float32)
    for i in range(64):
        c = min(max(z1[i], 0.0), 1.0)
        h1[i] = c
        h1[i + 64] = c * c

    h2 = np.empty(32, dtype=np.float32)
    for i in range(32):
        s = b_fc2[i]
        for j in range(128):
            s += w_fc2[i, j] * h1[j]
        c = min(max(s, 0.0), 1.0)
        h2[i] = c * c

    out = b_out[0]
    for i in range(32):
        out += w_out[i] * h2[i]

    return out


class Evaluator:
    def __init__(self, model_path="guitarfish_v3.gm"):
        self.metadata = {}
        self.wdl_scale = DEFAULT_WDL_SCALE
        self.sq_to_bucket = SQUARE_TO_BUCKET

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
                self.w_fc1 = np.ascontiguousarray(w["fc1_weight"], dtype=np.float32)
                self.b_fc1 = np.ascontiguousarray(w["fc1_bias"], dtype=np.float32)
                self.w_fc2 = np.ascontiguousarray(w["fc2_weight"], dtype=np.float32)
                self.b_fc2 = np.ascontiguousarray(w["fc2_bias"], dtype=np.float32)
                self.w_out = np.ascontiguousarray(
                    w["out_weight"].flatten(), dtype=np.float32
                )
                self.b_out = np.ascontiguousarray(
                    w["out_bias"].flatten(), dtype=np.float32
                )

        self.l1_size = self.ft_b.shape[0]

        self.w_acc_stack = np.zeros((MAX_PLY, self.l1_size), dtype=np.float32)
        self.b_acc_stack = np.zeros((MAX_PLY, self.l1_size), dtype=np.float32)
        self.w_bkt_stack = np.zeros(MAX_PLY, dtype=np.int64)
        self.b_bkt_stack = np.zeros(MAX_PLY, dtype=np.int64)
        self.hash_stack = np.zeros(MAX_PLY, dtype=np.uint64)
        self.score_stack = np.zeros(MAX_PLY, dtype=np.int32)

        self._warmup()

    def _warmup(self):
        dummy_acc = np.zeros(self.l1_size, dtype=np.float32)
        _forward(
            dummy_acc,
            dummy_acc,
            self.w_fc1,
            self.b_fc1,
            self.w_fc2,
            self.b_fc2,
            self.w_out,
            self.b_out,
        )

    def evaluate(self, board: chess.Board) -> int:
        ply = min(board._ply, MAX_PLY - 1)
        pos_hash = np.uint64(board._hash)

        if self.hash_stack[ply] == pos_hash:
            return int(self.score_stack[ply])

        can_inc = False
        if ply > 0 and self.hash_stack[ply - 1] == np.uint64(
            board._state_hash[ply - 1]
        ):
            m_val = int(board._state_moves[ply - 1])
            if m_val != -1:
                can_inc = True

        if can_inc:
            m_val = int(board._state_moves[ply - 1])
            mov_idx = int(board._state_moved_pt[ply - 1])
            cap_idx = int(board._state_captured_pt[ply - 1])
            new_w_bkt, new_b_bkt = _update_incremental(
                board._state_pieces[ply - 1],
                board._pieces,
                self.w_acc_stack[ply - 1],
                self.b_acc_stack[ply - 1],
                self.w_bkt_stack[ply - 1],
                self.b_bkt_stack[ply - 1],
                self.w_acc_stack[ply],
                self.b_acc_stack[ply],
                m_val,
                mov_idx,
                cap_idx,
                self.ft_w,
                self.ft_b,
                self.sq_to_bucket,
            )
            self.w_bkt_stack[ply] = new_w_bkt
            self.b_bkt_stack[ply] = new_b_bkt
        else:
            self.w_bkt_stack[ply] = _refresh_acc_white(
                board._pieces,
                self.ft_w,
                self.ft_b,
                self.w_acc_stack[ply],
                self.sq_to_bucket,
            )
            self.b_bkt_stack[ply] = _refresh_acc_black(
                board._pieces,
                self.ft_w,
                self.ft_b,
                self.b_acc_stack[ply],
                self.sq_to_bucket,
            )

        if board.turn:
            acc_stm = self.w_acc_stack[ply]
            acc_opp = self.b_acc_stack[ply]
        else:
            acc_stm = self.b_acc_stack[ply]
            acc_opp = self.w_acc_stack[ply]

        logit = _forward(
            acc_stm,
            acc_opp,
            self.w_fc1,
            self.b_fc1,
            self.w_fc2,
            self.b_fc2,
            self.w_out,
            self.b_out,
        )

        score = int(logit * self.wdl_scale)
        self.hash_stack[ply] = pos_hash
        self.score_stack[ply] = score
        return score
