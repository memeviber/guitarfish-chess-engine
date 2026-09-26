import io
import json
import os
import sys
import zipfile

import numpy as np
import onnxruntime as ort
from numba import njit

from src import bitboard as chess

DEFAULT_WDL_SCALE = 410.0


@njit(cache=True, fastmath=True)
def accumulate_features(out_acc, ft_w, ft_b, indices, count):
    for j in range(ft_b.shape[0]):
        out_acc[j] = ft_b[j]
    for i in range(count):
        row = ft_w[indices[i]]
        for j in range(ft_b.shape[0]):
            out_acc[j] += row[j]


class Evaluator:
    def __init__(self, model_path="guitarfish.gm"):
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

                onnx_bytes = z.read("mlp_int8.onnx")
                self.session = ort.InferenceSession(
                    onnx_bytes, sess_options=options, providers=["CPUExecutionProvider"]
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}")

        self.l1_size = self.ft_b.shape[0]
        self.buf_stm_indices = np.empty(256, dtype=np.int64)
        self.buf_opp_indices = np.empty(256, dtype=np.int64)
        self.acc_stm = np.empty(self.l1_size, dtype=np.float32)
        self.acc_opp = np.empty(self.l1_size, dtype=np.float32)
        self.onnx_input = np.empty((1, self.l1_size * 2), dtype=np.float32)

        self._warmup()

    def _warmup(self):
        chess.Board().perft(3)
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
