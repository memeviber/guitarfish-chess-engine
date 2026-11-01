import json
import urllib.parse
import urllib.request


class OnlineTablebase:
    def __init__(self, timeout=1.5):
        self.base_url = "https://tablebase.lichess.ovh/standard"
        self.timeout = timeout
        self.cache = {}

    def _pick_optimal_move(self, data):
        moves = data.get("moves", [])

        if not moves:
            return None

        best = moves[0]

        category = data.get("category", "unknown")
        dtm = best.get("dtm")
        dtz = best.get("dtz")

        if category == "win":
            if dtm is not None:
                score_cp = 20000 - abs(dtm)
            elif dtz is not None:
                score_cp = 19000 - abs(dtz)
            else:
                score_cp = 18000

        elif category == "loss":
            if dtm is not None:
                score_cp = -20000 + abs(dtm)
            elif dtz is not None:
                score_cp = -19000 + abs(dtz)
            else:
                score_cp = -18000

        elif category == "draw":
            score_cp = 0

        elif category in ("cursed-win", "maybe-win"):
            score_cp = 10000

        elif category in ("blessed-loss", "maybe-loss"):
            score_cp = -10000

        else:
            score_cp = 0

        return {
            "best_move": best["uci"],
            "category": category,
            "dtm": dtm,
            "dtz": dtz,
            "score_cp": score_cp,
        }

    def probe(self, fen: str):
        if fen in self.cache:
            return self.cache[fen]

        try:
            encoded_fen = urllib.parse.quote(fen, safe="")
            url = f"{self.base_url}?fen={encoded_fen}"

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Guitarfish-Engine/v6.4"},
            )

            with urllib.request.urlopen(
                req,
                timeout=self.timeout,
            ) as resp:
                if resp.status != 200:
                    return None

                data = json.loads(resp.read().decode("utf-8"))

            result = self._pick_optimal_move(data)

            if result is not None:
                self.cache[fen] = result

            return result

        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            return None


class OfflineTablebase:
    def __init__(self, path):
        try:
            import chess as python_chess
            import chess.syzygy  # Too lazy to implement some shit offline tablebase probing function in Numba,
        # so we're using python-chess's implementation :}}}
        except ImportError as exc:
            raise RuntimeError(
                "Offline Syzygy requires the python-chess package"
            ) from exc

        self.chess = python_chess
        self.tablebase = python_chess.syzygy.open_tablebase(path)
        self.cache = {}

    def probe(self, fen: str):
        if fen in self.cache:
            return self.cache[fen]

        try:
            board = self.chess.Board(fen)
            if len(board.piece_map()) > 7:
                return None

            best_move = None
            best_wdl = -3
            best_dtz = None
            best_rank = None

            for move in board.legal_moves:
                board.push(move)
                try:
                    wdl = -self.tablebase.probe_wdl(board)
                    try:
                        dtz = -self.tablebase.probe_dtz(board)
                    except KeyError:
                        dtz = None
                finally:
                    board.pop()

                if wdl > 0:
                    dtz_rank = -abs(dtz) if dtz is not None else 0
                elif wdl < 0:
                    dtz_rank = abs(dtz) if dtz is not None else 0
                else:
                    dtz_rank = 0
                rank = (wdl, dtz_rank)

                if best_rank is None or rank > best_rank:
                    best_move = move
                    best_wdl = wdl
                    best_dtz = dtz
                    best_rank = rank

            if best_move is None:
                return None

        except (KeyError, ValueError):
            return None

        categories = {
            2: "win",
            1: "cursed-win",
            0: "draw",
            -1: "blessed-loss",
            -2: "loss",
        }
        category = categories[best_wdl]
        if category == "win":
            score_cp = 19000 - abs(best_dtz) if best_dtz is not None else 18000
        elif category == "loss":
            score_cp = -19000 + abs(best_dtz) if best_dtz is not None else -18000
        elif category == "cursed-win":
            score_cp = 10000
        elif category == "blessed-loss":
            score_cp = -10000
        else:
            score_cp = 0

        result = {
            "best_move": best_move.uci(),
            "category": category,
            "dtm": None,
            "dtz": best_dtz,
            "score_cp": score_cp,
        }
        self.cache[fen] = result
        return result

    def close(self):
        self.tablebase.close()
