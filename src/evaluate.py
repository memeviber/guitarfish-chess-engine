import io
import json
import os
import weakref
import zipfile

import numpy as np
from numba import njit

from src import bitboard as chess

DEFAULT_WDL_SCALE = 410.0

SQUARE_TO_BUCKET = np.zeros(64, dtype=np.int64)
for sq in range(64):
    f = sq % 8
    r = sq // 8
    f_sym = f if f < 4 else 7 - f
    f_group = 0 if f_sym < 2 else 1
    r_group = r // 2
    SQUARE_TO_BUCKET[sq] = r_group * 2 + f_group


@njit(cache = True, fastmath=True)
def transform_features(raw_indices, count, sq_to_bucket, out_indices):
    k_bucket = 0
    for i in range(count):
        f = raw_indices[i]
        if 320 <= f < 384:
            k_bucket = sq_to_bucket[f - 320]
            break

    offset = k_bucket * 768
    for i in range(count):
        f = raw_indices[i]
        if f < 768:
            out_indices[i] = offset + f
        else:
            out_indices[i] = 6144 + f - 768


@njit(cache = True, fastmath=True)
def accumulate_features(out_acc, ft_w, ft_b, indices, count):
    for j in range(1024):
        out_acc[j] = ft_b[j]
    for i in range(count):
        idx = indices[i]
        for j in range(1024):
            out_acc[j] += ft_w[idx, j]


@njit(cache = True, fastmath=True)
def update_accumulator(out_acc, ft_w, old_indices, old_count, new_indices, new_count):
    old_i = 0
    new_i = 0
    while old_i < old_count and new_i < new_count:
        old_idx = old_indices[old_i]
        new_idx = new_indices[new_i]
        if old_idx < new_idx:
            for j in range(1024):
                out_acc[j] -= ft_w[old_idx, j]
            old_i += 1
        elif new_idx < old_idx:
            for j in range(1024):
                out_acc[j] += ft_w[new_idx, j]
            new_i += 1
        else:
            old_i += 1
            new_i += 1

    while old_i < old_count:
        old_idx = old_indices[old_i]
        for j in range(1024):
            out_acc[j] -= ft_w[old_idx, j]
        old_i += 1

    while new_i < new_count:
        new_idx = new_indices[new_i]
        for j in range(1024):
            out_acc[j] += ft_w[new_idx, j]
        new_i += 1


@njit(cache = True, fastmath=True)
def forward(acc_stm, acc_opp, w_fc1, b_fc1, w_fc2, b_fc2, w_out, b_out):
    pooled = np.empty(1024, dtype=np.float32)

    for i in range(512):
        p1 = min(max(acc_stm[i], 0.0), 1.0)
        p2 = min(max(acc_stm[i + 512], 0.0), 1.0)
        pooled[i] = p1 * p2

    for i in range(512):
        p1 = min(max(acc_opp[i], 0.0), 1.0)
        p2 = min(max(acc_opp[i + 512], 0.0), 1.0)
        pooled[i + 512] = p1 * p2

    z1 = np.empty(64, dtype=np.float32)
    for i in range(64):
        s = b_fc1[i]
        for j in range(1024):
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
    def __init__(self, model_path="guitarfish.gm"):
        self.metadata = {}
        self.wdl_scale = DEFAULT_WDL_SCALE
        self.sq_to_bucket = SQUARE_TO_BUCKET

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
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

        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}")

        self.buf_stm_indices = np.empty(256, dtype=np.int64)
        self.buf_opp_indices = np.empty(256, dtype=np.int64)
        self.trans_stm_indices = np.empty(256, dtype=np.int64)
        self.trans_opp_indices = np.empty(256, dtype=np.int64)

        self.acc_stm = np.empty(1024, dtype=np.float32)
        self.acc_opp = np.empty(1024, dtype=np.float32)
        self._board_states = weakref.WeakKeyDictionary()

        self._warmup()

    def _warmup(self):
        dummy_acc = np.zeros(1024, dtype=np.float32)
        dummy_indices = np.zeros(1, dtype=np.int64)
        accumulate_features(dummy_acc, self.ft_w, self.ft_b, dummy_indices, 1)
        forward(
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
        ply = board._ply
        states = self._board_states.get(board)
        if states is None:
            states = []
            self._board_states[board] = states
        while len(states) <= ply:
            states.append(
                {
                    "hash": None,
                    "white_features": np.empty(256, dtype=np.int64),
                    "black_features": np.empty(256, dtype=np.int64),
                    "white_count": 0,
                    "black_count": 0,
                    "white_acc": np.empty(1024, dtype=np.float32),
                    "black_acc": np.empty(1024, dtype=np.float32),
                    "score": 0,
                }
            )

        state = states[ply]
        position_hash = int(board._hash)
        if state["hash"] == position_hash:
            return state["score"]

        stm_cnt, opp_cnt = chess.extract_indices(
            board._pieces,
            board.turn,
            self.buf_stm_indices,
            self.buf_opp_indices,
        )

        transform_features(
            self.buf_stm_indices, stm_cnt, self.sq_to_bucket, self.trans_stm_indices
        )
        transform_features(
            self.buf_opp_indices, opp_cnt, self.sq_to_bucket, self.trans_opp_indices
        )

        if board.turn:
            white_features = self.trans_stm_indices
            white_count = stm_cnt
            black_features = self.trans_opp_indices
            black_count = opp_cnt
        else:
            white_features = self.trans_opp_indices
            white_count = opp_cnt
            black_features = self.trans_stm_indices
            black_count = stm_cnt

        white_features[:white_count].sort()
        black_features[:black_count].sort()

        parent = None
        if ply > 0:
            candidate = states[ply - 1]
            if candidate["hash"] == int(board._state_hash[ply - 1]):
                parent = candidate

        if parent is None:
            accumulate_features(
                state["white_acc"], self.ft_w, self.ft_b, white_features, white_count
            )
            accumulate_features(
                state["black_acc"], self.ft_w, self.ft_b, black_features, black_count
            )
        else:
            np.copyto(state["white_acc"], parent["white_acc"])
            np.copyto(state["black_acc"], parent["black_acc"])
            update_accumulator(
                state["white_acc"],
                self.ft_w,
                parent["white_features"],
                parent["white_count"],
                white_features,
                white_count,
            )
            update_accumulator(
                state["black_acc"],
                self.ft_w,
                parent["black_features"],
                parent["black_count"],
                black_features,
                black_count,
            )

        np.copyto(state["white_features"][:white_count], white_features[:white_count])
        np.copyto(state["black_features"][:black_count], black_features[:black_count])
        state["white_count"] = white_count
        state["black_count"] = black_count
        state["hash"] = position_hash

        if board.turn:
            self.acc_stm = state["white_acc"]
            self.acc_opp = state["black_acc"]
        else:
            self.acc_stm = state["black_acc"]
            self.acc_opp = state["white_acc"]

        logit = forward(
            self.acc_stm,
            self.acc_opp,
            self.w_fc1,
            self.b_fc1,
            self.w_fc2,
            self.b_fc2,
            self.w_out,
            self.b_out,
        )
        state["score"] = int(logit * self.wdl_scale)
        return state["score"]
