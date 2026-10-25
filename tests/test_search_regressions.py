import unittest

import chess as reference_chess
import src.bitboard as engine_chess
from src.search import MATE_VALUE, Searcher


PIECE_VALUES = (100, 320, 330, 500, 900, 0)


class MaterialEvaluator:
    def evaluate(self, board):
        white_score = sum(
            (1 if index < 6 else -1)
            * PIECE_VALUES[index % 6]
            * int(bitboard).bit_count()
            for index, bitboard in enumerate(board._pieces)
        )
        return white_score if board.turn else -white_score


class SearchRegressionTests(unittest.TestCase):
    def setUp(self):
        self.searcher = Searcher(MaterialEvaluator(), book_path="book.bin", tt_size_mb=1)
        self.searcher.own_book = False
        self.searcher.online_tablebase_enabled = False

    def tearDown(self):
        self.searcher.close()

    def test_finds_mate_in_one(self):
        board = engine_chess.Board("7k/5Q2/5K2/8/8/8/8/8 w - - 0 1")

        self.assertEqual(self.searcher.search(board, fixed_depth=3, silent=True), "f7g7")

    def test_captures_hanging_queen(self):
        board = engine_chess.Board("k7/8/8/3q4/4Q3/8/8/7K w - - 0 1")

        self.assertEqual(self.searcher.search(board, fixed_depth=3, silent=True), "e4d5")

    def test_promotes_pawn(self):
        board = engine_chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")

        self.assertEqual(self.searcher.search(board, fixed_depth=3, silent=True), "a7a8q")

    def test_quiescence_recognizes_stalemate_at_depth_limit(self):
        board = engine_chess.Board("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")

        self.assertEqual(self.searcher.quiescence(board, -MATE_VALUE, MATE_VALUE, 32), 0)

    def test_quiescence_preserves_checkmate_at_depth_limit(self):
        board = engine_chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1")

        self.assertEqual(
            self.searcher.quiescence(board, -MATE_VALUE, MATE_VALUE, 32),
            -MATE_VALUE + 32,
        )

    def test_null_move_is_disabled_in_minor_piece_and_near_rule_positions(self):
        minor_piece_endgame = engine_chess.Board(
            "4k3/nn6/8/8/8/8/NN6/PP2K3 w - - 0 1"
        )
        self.assertFalse(self.searcher._can_null_move(minor_piece_endgame))
        self.assertTrue(self.searcher._can_null_move(engine_chess.Board()))

        near_fifty_move_limit = engine_chess.Board()
        near_fifty_move_limit.halfmove_clock = 95
        self.assertFalse(self.searcher._can_null_move(near_fifty_move_limit))

    def test_quiet_move_to_en_passant_square_is_not_ordered_as_a_capture(self):
        board = engine_chess.Board("4k3/8/8/1N1pP3/8/8/8/4K3 w - d6 0 1")
        quiet_move = engine_chess.Move.from_uci("b5d6")
        en_passant_move = engine_chess.Move.from_uci("e5d6")
        self.assertIn(quiet_move, board.legal_moves)
        self.assertIn(en_passant_move, board.legal_moves)

        mailbox = self.searcher._update_mailbox(board._pieces, 0)
        scores = dict(
            (move.uci(), score)
            for score, move in self.searcher.score_moves_fast(
                board,
                [quiet_move, en_passant_move],
                1,
                0,
                None,
                mailbox,
            )
        )

        self.assertLess(scores["b5d6"], 900_000)
        self.assertGreaterEqual(scores["e5d6"], 1_000_000)

    def test_search_does_not_mutate_root_position(self):
        board = engine_chess.Board()
        pieces = board._pieces.copy()
        position_hash = int(board._hash)
        halfmove_clock = board.halfmove_clock

        move = self.searcher.search(board, fixed_depth=3, silent=True)

        self.assertIn(move, {candidate.uci() for candidate in board.legal_moves})
        self.assertTrue((board._pieces == pieces).all())
        self.assertEqual(int(board._hash), position_hash)
        self.assertEqual(board.halfmove_clock, halfmove_clock)

    def test_legal_moves_match_python_chess_in_special_positions(self):
        fens = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R "
            "w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 "
            "w kq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        )

        for fen in fens:
            with self.subTest(fen=fen):
                engine_board = engine_chess.Board(fen)
                reference_board = reference_chess.Board(fen)
                self.assertTrue(reference_board.is_valid())
                engine_moves = {move.uci() for move in engine_board.legal_moves}
                reference_moves = {move.uci() for move in reference_board.legal_moves}
                self.assertEqual(engine_moves, reference_moves)


if __name__ == "__main__":
    unittest.main()
