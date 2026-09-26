#!/usr/bin/env python3
import os
import sys

import src.uci as uci

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    try:
        model_file = sys.argv[1]
        book_file = sys.argv[2] if len(sys.argv) > 2 else ""
        uci.uci_loop(model_file, book_file)
    except KeyboardInterrupt:
        pass
    except IndexError:
        print("Usage: python guitarfish.py <model_file> <book_file>")
