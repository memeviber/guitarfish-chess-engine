import sys

import src.engine as engine


if __name__ == "__main__":
    try:
        model_file = sys.argv[1]
        book_file = sys.argv[2]
        engine.uci_loop(model_file, book_file)
    except KeyboardInterrupt:
        pass
    except IndexError:
        print("Usage: python guitarfish.py <model_file> <book_file>")
