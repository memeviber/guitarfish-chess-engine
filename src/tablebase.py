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
                headers={"User-Agent": "Guitarfish-Engine/6.1"},
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
