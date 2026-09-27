import io
import json
import os
import sys
import zipfile

from numba import njit
import numpy as np

from src import bitboard as chess

DEFAULT_WDL_SCALE = 410.0


@njit(fastmath=True, cache=True)
def accumulate_features(out_acc, ft_w, ft_b, indices, count):
    for j in range(1024):
        out_acc[j] = ft_b[j]
    for i in range(count):
        idx = indices[i]
        for j in range(1024):
            out_acc[j] += ft_w[idx, j]


@njit(fastmath=True, cache=True)
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
        self.acc_stm = np.empty(1024, dtype=np.float32)
        self.acc_opp = np.empty(1024, dtype=np.float32)

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
        return int(logit * self.wdl_scale)
