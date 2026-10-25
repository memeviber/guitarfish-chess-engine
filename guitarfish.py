#!/usr/bin/env python3
import os
import sys

import src.uci as uci

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    if len(sys.argv) > 4:
        print(
            "Usage: python guitarfish.py [eval_file] [book_file] [hash_mb]",
            file=sys.stderr,
        )
        sys.exit(2)

    bundle_dir = getattr(sys, "_MEIPASS", None)
    default_model = (
        os.path.join(bundle_dir, "guitarfish.gm")
        if bundle_dir
        else "guitarfish.gm"
    )
    model_file = sys.argv[1] if len(sys.argv) > 1 else default_model
    book_file = sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        hash_size_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        if not 1 <= hash_size_mb <= 1024:
            raise ValueError
    except ValueError:
        print("Error: hash_mb must be an integer from 1 to 1024", file=sys.stderr)
        sys.exit(2)

    try:
        uci.uci_loop(model_file, book_file, hash_size_mb)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
