import os
import struct

import numpy as np
from numba import njit

from src.polyglot_keys import POLYGLOT_KEYS

WHITE = True
BLACK = False
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)

A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)

_FILE_A = np.uint64(0x0101010101010101)
_FILE_B = np.uint64(0x0202020202020202)
_FILE_C = np.uint64(0x0404040404040404)
_FILE_D = np.uint64(0x0808080808080808)
_FILE_E = np.uint64(0x1010101010101010)
_FILE_F = np.uint64(0x2020202020202020)
_FILE_G = np.uint64(0x4040404040404040)
_FILE_H = np.uint64(0x8080808080808080)
_FILES = np.array(
    [_FILE_A, _FILE_B, _FILE_C, _FILE_D, _FILE_E, _FILE_F, _FILE_G, _FILE_H],
    dtype=np.uint64,
)

_PROMOTIONS = (KNIGHT, BISHOP, ROOK, QUEEN)
_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
_KING_ATTACKS = np.zeros(64, dtype=np.uint64)
_KING_ZONE_MASK = np.zeros(64, dtype=np.uint64)
_PASSED_WHITE = np.zeros(64, dtype=np.uint64)
_PASSED_BLACK = np.zeros(64, dtype=np.uint64)
_ADJACENT_FILES = np.zeros(8, dtype=np.uint64)

for f in range(8):
    adj = np.uint64(0)
    if f > 0:
        adj |= _FILES[f - 1]
    if f < 7:
        adj |= _FILES[f + 1]
    _ADJACENT_FILES[f] = adj

for _square in range(64):
    _rank = _square // 8
    _file = _square & 7
    for _dr, _df in (
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    ):
        _tr = _rank + _dr
        _tf = _file + _df
        if 0 <= _tr < 8 and 0 <= _tf < 8:
            _KNIGHT_ATTACKS[_square] |= np.uint64(1 << (_tr * 8 + _tf))
    for _dr in (-1, 0, 1):
        for _df in (-1, 0, 1):
            if _dr == 0 and _df == 0:
                continue
            _tr = _rank + _dr
            _tf = _file + _df
            if 0 <= _tr < 8 and 0 <= _tf < 8:
                _KING_ATTACKS[_square] |= np.uint64(1 << (_tr * 8 + _tf))
    _KING_ZONE_MASK[_square] = _KING_ATTACKS[_square] | (
        np.uint64(1) << np.uint64(_square)
    )

    file_span = _FILES[_file] | _ADJACENT_FILES[_file]
    w_span = np.uint64(0)
    for r_ahead in range(_rank + 1, 8):
        w_span |= np.uint64(0xFF << (r_ahead * 8)) & file_span
    _PASSED_WHITE[_square] = w_span

    b_span = np.uint64(0)
    for r_ahead in range(0, _rank):
        b_span |= np.uint64(0xFF << (r_ahead * 8)) & file_span
    _PASSED_BLACK[_square] = b_span

# fmt: off
_DEBRUIJN_INDEX = np.array(
    (
         0,  1, 48,  2, 57, 49, 28,  3,
        61, 58, 50, 42, 38, 29, 17,  4,
        62, 55, 59, 36, 53, 51, 43, 22,
        45, 39, 33, 30, 24, 18, 12,  5,
        63, 47, 56, 27, 60, 41, 37, 16,
        54, 35, 52, 21, 44, 32, 23, 11,
        46, 26, 40, 15, 34, 20, 31, 10,
        25, 14, 19,  9, 13,  8,  7,  6,
    ),
    dtype=np.int64,
)
# fmt: on


@njit(cache=True, inline="always", fastmath=True)
def _lsb_index(bitboard):
    index = (bitboard * _DEBRUIJN) >> np.uint64(58)
    return _DEBRUIJN_INDEX[index]


@njit(cache=True, inline="always", fastmath=True)
def _pop_lsb(bitboard):
    bit = bitboard & (np.uint64(0) - bitboard)
    square = _lsb_index(bit)
    return bit, square


@njit(cache=True, inline="always", fastmath=True)
def _ray_attacks(square, occupied, dr, df):
    rank = square // 8
    file = square & 7
    attacks = np.uint64(0)
    rank += dr
    file += df
    while 0 <= rank < 8 and 0 <= file < 8:
        target = np.uint64(1) << np.uint64(rank * 8 + file)
        attacks |= target
        if occupied & target:
            break
        rank += dr
        file += df
    return attacks


@njit(cache=True, inline="always", fastmath=True)
def _bishop_attacks(square, occupied):
    attacks = np.uint64(0)
    for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        attacks |= _ray_attacks(square, occupied, dr, df)
    return attacks


@njit(cache=True, inline="always", fastmath=True)
def _rook_attacks(square, occupied):
    attacks = np.uint64(0)
    for dr, df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        attacks |= _ray_attacks(square, occupied, dr, df)
    return attacks


@njit(cache=True, inline="always", fastmath=True)
def _queen_attacks(square, occupied):
    return _bishop_attacks(square, occupied) | _rook_attacks(square, occupied)


@njit(cache=True, fastmath=True)
def get_color_attacks(pieces, is_white):
    offset = 0 if is_white else 6
    occupied = np.uint64(0)
    for i in range(12):
        occupied |= pieces[i]

    attacks = np.uint64(0)
    pawns = pieces[offset]
    if is_white:
        attacks |= ((pawns << np.uint64(7)) & ~_FILE_H) | (
            (pawns << np.uint64(9)) & ~_FILE_A
        )
    else:
        attacks |= ((pawns >> np.uint64(7)) & ~_FILE_A) | (
            (pawns >> np.uint64(9)) & ~_FILE_H
        )

    knights = pieces[offset + 1]
    while knights:
        bit, sq = _pop_lsb(knights)
        attacks |= _KNIGHT_ATTACKS[sq]
        knights ^= bit

    bishops = pieces[offset + 2]
    while bishops:
        bit, sq = _pop_lsb(bishops)
        attacks |= _bishop_attacks(sq, occupied)
        bishops ^= bit

    rooks = pieces[offset + 3]
    while rooks:
        bit, sq = _pop_lsb(rooks)
        attacks |= _rook_attacks(sq, occupied)
        rooks ^= bit

    queens = pieces[offset + 4]
    while queens:
        bit, sq = _pop_lsb(queens)
        attacks |= _queen_attacks(sq, occupied)
        queens ^= bit

    kings = pieces[offset + 5]
    if kings:
        _, sq = _pop_lsb(kings)
        attacks |= _KING_ATTACKS[sq]

    return attacks


@njit(cache=True, fastmath=True)
def extract_indices(pieces, turn_white, out_stm, out_opp):
    w_count = 0
    b_count = 0
    w_idx = np.empty(256, dtype=np.int64)
    b_idx = np.empty(256, dtype=np.int64)

    for p_type in range(6):
        bb = pieces[p_type]
        while bb:
            bit, sq = _pop_lsb(bb)
            w_idx[w_count] = p_type * 64 + sq
            w_count += 1
            b_idx[b_count] = (p_type + 6) * 64 + (sq ^ 56)
            b_count += 1
            bb ^= bit

        bb = pieces[p_type + 6]
        while bb:
            bit, sq = _pop_lsb(bb)
            b_idx[b_count] = p_type * 64 + (sq ^ 56)
            b_count += 1
            w_idx[w_count] = (p_type + 6) * 64 + sq
            w_count += 1
            bb ^= bit

    w_attacks = get_color_attacks(pieces, True)
    b_attacks = get_color_attacks(pieces, False)

    w_occ = pieces[0] | pieces[1] | pieces[2] | pieces[3] | pieces[4] | pieces[5]
    b_occ = pieces[6] | pieces[7] | pieces[8] | pieces[9] | pieces[10] | pieces[11]

    contested = w_attacks & b_attacks
    while contested:
        bit, sq = _pop_lsb(contested)
        w_idx[w_count] = 768 + sq
        w_count += 1
        b_idx[b_count] = 768 + (sq ^ 56)
        b_count += 1
        contested ^= bit

    w_attacked = w_occ & b_attacks
    b_attacked = b_occ & w_attacks
    w_hanging = w_attacked & ~w_attacks
    b_hanging = b_attacked & ~b_attacks

    bb = w_attacked
    while bb:
        bit, sq = _pop_lsb(bb)
        w_idx[w_count] = 832 + sq
        w_count += 1
        b_idx[b_count] = 896 + (sq ^ 56)
        b_count += 1
        bb ^= bit

    bb = b_attacked
    while bb:
        bit, sq = _pop_lsb(bb)
        w_idx[w_count] = 896 + sq
        w_count += 1
        b_idx[b_count] = 832 + (sq ^ 56)
        b_count += 1
        bb ^= bit

    bb = w_hanging
    while bb:
        bit, sq = _pop_lsb(bb)
        w_idx[w_count] = 960 + sq
        w_count += 1
        b_idx[b_count] = 1024 + (sq ^ 56)
        b_count += 1
        bb ^= bit

    bb = b_hanging
    while bb:
        bit, sq = _pop_lsb(bb)
        w_idx[w_count] = 1024 + sq
        w_count += 1
        b_idx[b_count] = 960 + (sq ^ 56)
        b_count += 1
        bb ^= bit

    if pieces[5]:
        _, wk_sq = _pop_lsb(pieces[5])
        bb = b_attacks & _KING_ZONE_MASK[wk_sq]
        while bb:
            bit, sq = _pop_lsb(bb)
            w_idx[w_count] = 1088 + sq
            w_count += 1
            b_idx[b_count] = 1152 + (sq ^ 56)
            b_count += 1
            bb ^= bit
        if b_attacks & (np.uint64(1) << np.uint64(wk_sq)):
            w_idx[w_count] = 1216
            w_count += 1

    if pieces[11]:
        _, bk_sq = _pop_lsb(pieces[11])
        bb = w_attacks & _KING_ZONE_MASK[bk_sq]
        while bb:
            bit, sq = _pop_lsb(bb)
            b_idx[b_count] = 1088 + (sq ^ 56)
            b_count += 1
            w_idx[w_count] = 1152 + sq
            w_count += 1
            bb ^= bit
        if w_attacks & (np.uint64(1) << np.uint64(bk_sq)):
            b_idx[b_count] = 1216
            b_count += 1

    w_pawns = pieces[0]
    b_pawns = pieces[6]

    bb = w_pawns
    while bb:
        bit, sq = _pop_lsb(bb)
        if not (b_pawns & _PASSED_WHITE[sq]):
            w_idx[w_count] = 1217 + sq
            w_count += 1
            b_idx[b_count] = 1281 + (sq ^ 56)
            b_count += 1
        bb ^= bit

    bb = b_pawns
    while bb:
        bit, sq = _pop_lsb(bb)
        if not (w_pawns & _PASSED_BLACK[sq]):
            b_idx[b_count] = 1217 + (sq ^ 56)
            b_count += 1
            w_idx[w_count] = 1281 + sq
            w_count += 1
        bb ^= bit

    for f in range(8):
        f_mask = _FILES[f]
        adj_mask = _ADJACENT_FILES[f]
        wp_f = w_pawns & f_mask
        if wp_f:
            if not (w_pawns & adj_mask):
                bb = wp_f
                while bb:
                    bit, sq = _pop_lsb(bb)
                    w_idx[w_count] = 1345 + sq
                    w_count += 1
                    b_idx[b_count] = 1409 + (sq ^ 56)
                    b_count += 1
                    bb ^= bit
            if wp_f & (wp_f - np.uint64(1)):
                bb = wp_f
                while bb:
                    bit, sq = _pop_lsb(bb)
                    w_idx[w_count] = 1473 + sq
                    w_count += 1
                    b_idx[b_count] = 1537 + (sq ^ 56)
                    b_count += 1
                    bb ^= bit

        bp_f = b_pawns & f_mask
        if bp_f:
            if not (b_pawns & adj_mask):
                bb = bp_f
                while bb:
                    bit, sq = _pop_lsb(bb)
                    b_idx[b_count] = 1345 + (sq ^ 56)
                    b_count += 1
                    w_idx[w_count] = 1409 + sq
                    w_count += 1
                    bb ^= bit
            if bp_f & (bp_f - np.uint64(1)):
                bb = bp_f
                while bb:
                    bit, sq = _pop_lsb(bb)
                    b_idx[b_count] = 1473 + (sq ^ 56)
                    b_count += 1
                    w_idx[w_count] = 1537 + sq
                    w_count += 1
                    bb ^= bit

    w_pattacks = ((w_pawns & ~_FILE_H) << np.uint64(9)) | (
        (w_pawns & ~_FILE_A) << np.uint64(7)
    )
    w_phalanx = ((w_pawns & ~_FILE_H) << np.uint64(1)) | (
        (w_pawns & ~_FILE_A) >> np.uint64(1)
    )
    bb = w_pawns & (w_pattacks | w_phalanx)
    while bb:
        bit, sq = _pop_lsb(bb)
        w_idx[w_count] = 1601 + sq
        w_count += 1
        b_idx[b_count] = 1665 + (sq ^ 56)
        b_count += 1
        bb ^= bit

    b_pattacks = ((b_pawns & ~_FILE_A) >> np.uint64(9)) | (
        (b_pawns & ~_FILE_H) >> np.uint64(7)
    )
    b_phalanx = ((b_pawns & ~_FILE_A) >> np.uint64(1)) | (
        (b_pawns & ~_FILE_H) << np.uint64(1)
    )
    bb = b_pawns & (b_pattacks | b_phalanx)
    while bb:
        bit, sq = _pop_lsb(bb)
        b_idx[b_count] = 1601 + (sq ^ 56)
        b_count += 1
        w_idx[w_count] = 1665 + sq
        w_count += 1
        bb ^= bit

    if turn_white:
        for i in range(w_count):
            out_stm[i] = w_idx[i]
        for i in range(b_count):
            out_opp[i] = b_idx[i]
        return w_count, b_count
    else:
        for i in range(b_count):
            out_stm[i] = b_idx[i]
        for i in range(w_count):
            out_opp[i] = w_idx[i]
        return b_count, w_count


@njit(cache=True, fastmath=True)
def is_attacked(square, by_white, pieces):
    occupied = np.uint64(0)
    for i in range(12):
        occupied |= pieces[i]
    pawn_index = 0 if by_white else 6
    pawns = pieces[pawn_index]
    target = np.uint64(1) << np.uint64(square)
    if by_white:
        if ((target >> np.uint64(7)) & ~_FILE_A) & pawns:
            return True
        if ((target >> np.uint64(9)) & ~_FILE_H) & pawns:
            return True
    else:
        if ((target << np.uint64(7)) & ~_FILE_H) & pawns:
            return True
        if ((target << np.uint64(9)) & ~_FILE_A) & pawns:
            return True

    knight_index = 1 if by_white else 7
    if _KNIGHT_ATTACKS[square] & pieces[knight_index]:
        return True
    king_index = 5 if by_white else 11
    if _KING_ATTACKS[square] & pieces[king_index]:
        return True

    bishop_index = 2 if by_white else 8
    rook_index = 3 if by_white else 9
    queen_index = 4 if by_white else 10
    for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        attacks = _ray_attacks(square, occupied, dr, df)
        if attacks & (pieces[bishop_index] | pieces[queen_index]):
            return True
    for dr, df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        attacks = _ray_attacks(square, occupied, dr, df)
        if attacks & (pieces[rook_index] | pieces[queen_index]):
            return True
    return False


@njit(cache=True, fastmath=True)
def _generate_pseudo_into(pieces, white_to_move, castling, ep_square, moves):
    count = 0
    own_start = 0 if white_to_move else 6
    enemy_start = 6 if white_to_move else 0
    occupied = np.uint64(0)
    own = np.uint64(0)
    enemy = np.uint64(0)
    for i in range(6):
        own |= pieces[own_start + i]
        enemy |= pieces[enemy_start + i]
        occupied |= pieces[own_start + i] | pieces[enemy_start + i]
    empty = ~occupied

    pawns = pieces[own_start]
    while pawns:
        bit, source = _pop_lsb(pawns)
        pawns ^= bit
        rank = source // 8
        file = source & 7
        step = 8 if white_to_move else -8
        target = source + step
        if 0 <= target < 64 and (empty & (np.uint64(1) << np.uint64(target))):
            if target // 8 in (0, 7):
                for promotion in _PROMOTIONS:
                    moves[count] = source | (target << 6) | (promotion << 12)
                    count += 1
            else:
                moves[count] = source | (target << 6)
                count += 1
                if (rank == 1 and white_to_move) or (rank == 6 and not white_to_move):
                    target2 = source + step * 2
                    if empty & (np.uint64(1) << np.uint64(target2)):
                        moves[count] = source | (target2 << 6)
                        count += 1
        for df in (-1, 1):
            target_file = file + df
            target = source + step + df
            if not (0 <= target_file < 8 and 0 <= target < 64):
                continue
            target_bit = np.uint64(1) << np.uint64(target)
            if (enemy & target_bit) or target == ep_square:
                if target // 8 in (0, 7):
                    for promotion in _PROMOTIONS:
                        moves[count] = source | (target << 6) | (promotion << 12)
                        count += 1
                else:
                    moves[count] = source | (target << 6)
                    count += 1

    for piece_type in (KNIGHT, KING):
        pieceset = pieces[own_start + piece_type - 1]
        while pieceset:
            bit, source = _pop_lsb(pieceset)
            pieceset ^= bit
            attacks = (
                _KNIGHT_ATTACKS[source]
                if piece_type == KNIGHT
                else _KING_ATTACKS[source]
            ) & ~own
            while attacks:
                target_bit, target = _pop_lsb(attacks)
                attacks ^= target_bit
                moves[count] = source | (target << 6)
                count += 1

    for piece_type in (BISHOP, ROOK, QUEEN):
        pieceset = pieces[own_start + piece_type - 1]
        direction_rows = (
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        )
        direction_count = 4 if piece_type == BISHOP else 4 if piece_type == ROOK else 8
        while pieceset:
            bit, source = _pop_lsb(pieceset)
            pieceset ^= bit
            start = 0 if piece_type != ROOK else 4
            for direction_index in range(start, start + direction_count):
                dr, df = direction_rows[direction_index]
                attacks = _ray_attacks(source, occupied, dr, df)
                while attacks:
                    target_bit, target = _pop_lsb(attacks)
                    attacks ^= target_bit
                    if not (own & target_bit):
                        moves[count] = source | (target << 6)
                        count += 1

    king = pieces[own_start + KING - 1]
    if king:
        _, source = _pop_lsb(king)
        rank = 0 if white_to_move else 7
        if white_to_move and source == E1 or not white_to_move and source == E8:
            rook_index = own_start + ROOK - 1
            king_rook = 7 if white_to_move else 63
            queen_rook = 0 if white_to_move else 56
            if castling & (1 if white_to_move else 4) and pieces[rook_index] & (
                np.uint64(1) << np.uint64(king_rook)
            ):
                if not (
                    occupied
                    & (
                        np.uint64(1) << np.uint64(rank * 8 + 5)
                        | np.uint64(1) << np.uint64(rank * 8 + 6)
                    )
                ):
                    moves[count] = source | ((rank * 8 + 6) << 6)
                    count += 1
            if castling & (2 if white_to_move else 8) and pieces[rook_index] & (
                np.uint64(1) << np.uint64(queen_rook)
            ):
                if not (
                    occupied
                    & (
                        np.uint64(1) << np.uint64(rank * 8 + 1)
                        | np.uint64(1) << np.uint64(rank * 8 + 2)
                        | np.uint64(1) << np.uint64(rank * 8 + 3)
                    )
                ):
                    moves[count] = source | ((rank * 8 + 2) << 6)
                    count += 1
    return moves, count


@njit(cache=True, fastmath=True)
def is_legal_encoded(pieces, white_to_move, ep_square, value):
    test_pieces = np.empty(12, dtype=np.uint64)
    for i in range(12):
        test_pieces[i] = pieces[i]

    source = value & 63
    target = (value >> 6) & 63
    promotion = (value >> 12) & 7
    source_bit = np.uint64(1) << np.uint64(source)
    target_bit = np.uint64(1) << np.uint64(target)
    own_start = 0 if white_to_move else 6
    moving_index = -1
    for i in range(6):
        if test_pieces[own_start + i] & source_bit:
            moving_index = own_start + i
            break
    if moving_index < 0:
        return False

    moving_type = moving_index - own_start + 1
    test_pieces[moving_index] &= ~source_bit
    for i in range(6):
        enemy_index = (6 if white_to_move else 0) + i
        test_pieces[enemy_index] &= ~target_bit

    is_ep = (
        moving_type == PAWN
        and target == ep_square
        and not (pieces[0 if white_to_move else 6] & target_bit)
    )
    if is_ep:
        capture_square = target - 8 if white_to_move else target + 8
        test_pieces[(6 if white_to_move else 0)] &= ~(
            np.uint64(1) << np.uint64(capture_square)
        )

    placed_index = (own_start + promotion - 1) if promotion else moving_index
    test_pieces[placed_index] |= target_bit

    if moving_type == KING and abs(target - source) == 2:
        rook_from = 7 if target > source else 0
        rook_to = 5 if target > source else 3
        rook_index = own_start + ROOK - 1
        test_pieces[rook_index] &= ~(
            np.uint64(1) << np.uint64(rook_from + (56 if not white_to_move else 0))
        )
        test_pieces[rook_index] |= np.uint64(1) << np.uint64(
            rook_to + (56 if not white_to_move else 0)
        )

    king_bits = test_pieces[own_start + KING - 1]
    if not king_bits:
        return False
    king_square = _lsb_index(king_bits)
    return not is_attacked(king_square, not white_to_move, test_pieces)


@njit(cache=True, fastmath=True)
def _generate_legal_into(pieces, white_to_move, castling, ep_square, pseudo, legal):
    pseudo, pseudo_count = _generate_pseudo_into(
        pieces, white_to_move, castling, ep_square, pseudo
    )
    count = 0
    for index in range(pseudo_count):
        value = pseudo[index]
        source = value & 63
        target = (value >> 6) & 63
        moving_index = (KING - 1) + (0 if white_to_move else 6)
        source_bit = np.uint64(1) << np.uint64(source)
        is_castle = bool(
            pieces[moving_index] & source_bit and abs(target - source) == 2
        )
        if is_castle:
            if is_attacked(source, not white_to_move, pieces):
                continue
            transit = source | (((source + target) // 2) << 6)
            if not is_legal_encoded(pieces, white_to_move, ep_square, transit):
                continue
        if is_legal_encoded(pieces, white_to_move, ep_square, value):
            legal[count] = value
            count += 1
    return legal, count


_SEE_PIECE_VALUES = np.array([100, 300, 300, 500, 900, 20000], dtype=np.int64)


@njit(cache=True, fastmath=True)
def see(pieces, white_to_move, from_sq, to_sq):
    target_bit = np.uint64(1) << np.uint64(to_sq)
    from_bit = np.uint64(1) << np.uint64(from_sq)

    victim_val = 0
    victim_type = -1
    enemy_offset = 6 if white_to_move else 0
    for pt in range(6):
        if pieces[enemy_offset + pt] & target_bit:
            victim_type = pt
            victim_val = _SEE_PIECE_VALUES[pt]
            break

    own_offset = 0 if white_to_move else 6
    moving_type = -1
    for pt in range(6):
        if pieces[own_offset + pt] & from_bit:
            moving_type = pt
            break

    if moving_type == -1:
        return 0

    occupied = np.uint64(0)
    for i in range(12):
        occupied |= pieces[i]

    if victim_type == -1 and moving_type == 0 and (from_sq & 7) != (to_sq & 7):
        victim_val = _SEE_PIECE_VALUES[0]
        cap_sq = to_sq - 8 if white_to_move else to_sq + 8
        occupied &= ~(np.uint64(1) << np.uint64(cap_sq))

    gain = np.empty(32, dtype=np.int64)
    d = 0
    gain[0] = victim_val
    curr_attacker_val = _SEE_PIECE_VALUES[moving_type]

    occupied &= ~from_bit
    side = not white_to_move

    bishops_queens = pieces[2] | pieces[8] | pieces[4] | pieces[10]
    rooks_queens = pieces[3] | pieces[9] | pieces[4] | pieces[10]

    w_pawns_att = (
        ((target_bit >> np.uint64(7)) & ~_FILE_A)
        | ((target_bit >> np.uint64(9)) & ~_FILE_H)
    ) & pieces[0]
    b_pawns_att = (
        ((target_bit << np.uint64(7)) & ~_FILE_H)
        | ((target_bit << np.uint64(9)) & ~_FILE_A)
    ) & pieces[6]
    knights_att = _KNIGHT_ATTACKS[to_sq] & (pieces[1] | pieces[7])
    kings_att = _KING_ATTACKS[to_sq] & (pieces[5] | pieces[11])
    bishops_att = _bishop_attacks(to_sq, occupied) & bishops_queens
    rooks_att = _rook_attacks(to_sq, occupied) & rooks_queens

    attackers = (
        w_pawns_att | b_pawns_att | knights_att | kings_att | bishops_att | rooks_att
    )

    while True:
        d += 1
        gain[d] = curr_attacker_val - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < 0:
            break

        cur_offset = 0 if side else 6
        opp_offset = 6 if side else 0

        chosen_bit = np.uint64(0)
        chosen_type = -1

        for pt in range(6):
            subset = attackers & pieces[cur_offset + pt] & occupied
            if subset:
                chosen_bit = subset & (np.uint64(0) - subset)
                chosen_type = pt
                break

        if chosen_type == -1:
            break

        if chosen_type == 5:
            opp_attackers = np.uint64(0)
            for pt in range(6):
                opp_attackers |= attackers & pieces[opp_offset + pt] & occupied
            if opp_attackers:
                break

        curr_attacker_val = _SEE_PIECE_VALUES[chosen_type]
        occupied &= ~chosen_bit

        if chosen_type in (0, 2, 3, 4):
            attackers |= _bishop_attacks(to_sq, occupied) & bishops_queens
            attackers |= _rook_attacks(to_sq, occupied) & rooks_queens
        attackers &= occupied

        side = not side

    while d > 1:
        d -= 1
        gain[d - 1] = -max(-gain[d - 1], gain[d])

    return gain[0]


@njit(cache=True, fastmath=True)
def fast_see(pieces, white_to_move, from_sq, to_sq, threshold=0):
    return see(pieces, white_to_move, from_sq, to_sq) >= threshold


@njit(cache=True, fastmath=True)
def _apply_encoded(pieces, white_to_move, castling, ep_square, value):
    source = value & 63
    target = (value >> 6) & 63
    promotion = (value >> 12) & 7
    own_start = 0 if white_to_move else 6
    enemy_start = 6 if white_to_move else 0
    source_bit = np.uint64(1) << np.uint64(source)
    target_bit = np.uint64(1) << np.uint64(target)

    moving_index = own_start
    for index in range(6):
        if pieces[own_start + index] & source_bit:
            moving_index = own_start + index
            break
    moving_type = moving_index - own_start + 1
    pieces[moving_index] &= ~source_bit

    captured_index = -1
    for index in range(6):
        if pieces[enemy_start + index] & target_bit:
            captured_index = enemy_start + index
            pieces[captured_index] &= ~target_bit
            break

    if moving_type == PAWN and target == ep_square and captured_index < 0:
        capture_square = target - 8 if white_to_move else target + 8
        pieces[enemy_start] &= ~(np.uint64(1) << np.uint64(capture_square))

    placed_index = own_start + promotion - 1 if promotion else moving_index
    pieces[placed_index] |= target_bit

    if moving_type == KING and abs(target - source) == 2:
        rook_from = 7 if target > source else 0
        rook_to = 5 if target > source else 3
        rank_offset = 0 if white_to_move else 56
        pieces[own_start + ROOK - 1] &= ~(
            np.uint64(1) << np.uint64(rook_from + rank_offset)
        )
        pieces[own_start + ROOK - 1] |= np.uint64(1) << np.uint64(rook_to + rank_offset)

    if moving_type == KING:
        castling &= ~(3 if white_to_move else 12)
    elif moving_type == ROOK:
        if source == A1:
            castling &= ~2
        elif source == H1:
            castling &= ~1
        elif source == A8:
            castling &= ~8
        elif source == H8:
            castling &= ~4

    if captured_index == 3 + enemy_start:
        if target == A1:
            castling &= ~2
        elif target == H1:
            castling &= ~1
        elif target == A8:
            castling &= ~8
        elif target == H8:
            castling &= ~4

    next_ep = -1
    if moving_type == PAWN and abs(target - source) == 16:
        next_ep = (source + target) // 2
    return not white_to_move, castling, next_ep


@njit(fastmath=True)
def _perft_stack(
    pieces,
    white_to_move,
    castling,
    ep_square,
    depth,
    piece_stack,
    pseudo_stack,
    move_stack,
    ply,
):
    if depth == 0:
        return 1
    moves, count = _generate_legal_into(
        pieces,
        white_to_move,
        castling,
        ep_square,
        pseudo_stack[ply],
        move_stack[ply],
    )
    nodes = 0
    for index in range(count):
        for p_idx in range(12):
            piece_stack[ply + 1, p_idx] = pieces[p_idx]
        child_turn, child_castling, child_ep = _apply_encoded(
            piece_stack[ply + 1],
            white_to_move,
            castling,
            ep_square,
            moves[index],
        )
        nodes += _perft_stack(
            piece_stack[ply + 1],
            child_turn,
            child_castling,
            child_ep,
            depth - 1,
            piece_stack,
            pseudo_stack,
            move_stack,
            ply + 1,
        )
    return nodes


@njit(fastmath=True)
def _perft_encoded(pieces, white_to_move, castling, ep_square, depth):
    piece_stack = np.empty((depth + 1, 12), dtype=np.uint64)
    pseudo_stack = np.empty((depth + 1, 256), dtype=np.int64)
    move_stack = np.empty((depth + 1, 256), dtype=np.int64)
    for p_idx in range(12):
        piece_stack[0, p_idx] = pieces[p_idx]
    return _perft_stack(
        piece_stack[0],
        white_to_move,
        castling,
        ep_square,
        depth,
        piece_stack,
        pseudo_stack,
        move_stack,
        0,
    )


@njit(cache=True, fastmath=True)
def _polyglot_hash(pieces, castling_rights, ep_square, turn_white, poly_keys):
    h = np.uint64(0)

    for p_type in range(6):
        w_bb = pieces[p_type]
        while w_bb:
            bit = w_bb & (np.uint64(0) - w_bb)
            sq = _lsb_index(bit)
            h ^= poly_keys[(p_type * 2 + 1) * 64 + sq]
            w_bb ^= bit

        b_bb = pieces[p_type + 6]
        while b_bb:
            bit = b_bb & (np.uint64(0) - b_bb)
            sq = _lsb_index(bit)
            h ^= poly_keys[(p_type * 2) * 64 + sq]
            b_bb ^= bit

    if castling_rights & 1:
        h ^= poly_keys[768]
    if castling_rights & 2:
        h ^= poly_keys[769]
    if castling_rights & 4:
        h ^= poly_keys[770]
    if castling_rights & 8:
        h ^= poly_keys[771]

    if ep_square >= 0:
        ep_file = ep_square & 7
        pawn_mask = pieces[0 if not turn_white else 6]
        p_row = ep_square - 8 if turn_white else ep_square + 8
        can_ep = False
        if ep_file > 0 and (pawn_mask & (np.uint64(1) << np.uint64(p_row - 1))):
            can_ep = True
        if ep_file < 7 and (pawn_mask & (np.uint64(1) << np.uint64(p_row + 1))):
            can_ep = True
        if can_ep:
            h ^= poly_keys[772 + ep_file]

    if turn_white:
        h ^= poly_keys[780]

    return int(h)


class PolyglotBook:
    def __init__(self, book_path="book.bin"):
        self.book_path = book_path
        self.is_loaded = os.path.exists(book_path)

    def probe(self, board):
        if not self.is_loaded or not os.path.exists(self.book_path):
            return None

        key = polyglot.zobrist_hash(board)
        size = os.path.getsize(self.book_path)
        num_entries = size // 16

        matching_moves = []
        matching_weights = []

        with open(self.book_path, "rb") as f:
            low, high = 0, num_entries - 1
            first_idx = -1

            while low <= high:
                mid = (low + high) // 2
                f.seek(mid * 16)
                entry_key, raw_move, weight, _ = struct.unpack(">QHHI", f.read(16))

                if entry_key >= key:
                    if entry_key == key:
                        first_idx = mid
                    high = mid - 1
                else:
                    low = mid + 1

            if first_idx != -1:
                idx = first_idx
                while idx < num_entries:
                    f.seek(idx * 16)
                    entry_key, raw_move, weight, _ = struct.unpack(">QHHI", f.read(16))
                    if entry_key != key:
                        break

                    to_f = raw_move & 7
                    to_r = (raw_move >> 3) & 7
                    from_f = (raw_move >> 6) & 7
                    from_r = (raw_move >> 9) & 7
                    promo_code = (raw_move >> 12) & 7

                    from_sq = from_r * 8 + from_f
                    to_sq = to_r * 8 + to_f
                    promo = (
                        {1: KNIGHT, 2: BISHOP, 3: ROOK, 4: QUEEN}.get(promo_code, None)
                        if promo_code
                        else None
                    )

                    move = Move(from_sq, to_sq, promo)
                    if board.is_legal(move):
                        matching_moves.append(move)
                        matching_weights.append(max(1, weight))
                    idx += 1

        if matching_moves:
            total_w = sum(matching_weights)
            r = np.random.randint(0, total_w)
            curr = 0
            for m, w in zip(matching_moves, matching_weights):
                curr += w
                if r < curr:
                    return m
            return matching_moves[0]
        return None


class _PolyglotNamespace:
    BookReader = PolyglotBook

    @staticmethod
    def zobrist_hash(board):
        return _polyglot_hash(
            board._pieces,
            board.castling_rights,
            board.ep_square,
            board.turn,
            POLYGLOT_KEYS,
        )

    @staticmethod
    def open_reader(book_path="book.bin"):
        return PolyglotBook(book_path)


polyglot = _PolyglotNamespace()


class Move:
    __slots__ = ("from_square", "to_square", "promotion", "_value")

    def __init__(self, from_square=0, to_square=0, promotion=None, value=None):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion
        self._value = (
            value
            if value is not None
            else from_square | (to_square << 6) | ((promotion or 0) << 12)
        )

    @classmethod
    def from_uci(cls, text):
        source = (ord(text[0]) - 97) + (ord(text[1]) - 49) * 8
        target = (ord(text[2]) - 97) + (ord(text[3]) - 49) * 8
        promotion = None
        if len(text) == 5:
            promotion = {"n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN}[text[4]]
        return cls(source, target, promotion)

    @staticmethod
    def null():
        return _NULL_MOVE

    @classmethod
    def from_value(cls, value):
        promotion = (value >> 12) & 7
        return cls(value & 63, (value >> 6) & 63, promotion or None, value)

    def uci(self):
        text = chr(97 + (self.from_square & 7)) + chr(49 + self.from_square // 8)
        text += chr(97 + (self.to_square & 7)) + chr(49 + self.to_square // 8)
        if self.promotion:
            text += {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}[self.promotion]
        return text

    def __hash__(self):
        return self._value

    def __eq__(self, other):
        return isinstance(other, Move) and self._value == other._value

    def __bool__(self):
        return True


_NULL_MOVE = Move(0, 0, value=-1)


class Piece:
    __slots__ = ("piece_type", "color")

    def __init__(self, piece_type, color):
        self.piece_type = piece_type
        self.color = color


class Board:
    def __init__(self, fen=None):
        MAX_HISTORY = 2048
        self._pieces = np.zeros(12, dtype=np.uint64)
        self._state_pieces = np.zeros((MAX_HISTORY, 12), dtype=np.uint64)
        self._state_turn = np.zeros(MAX_HISTORY, dtype=np.bool_)
        self._state_castling = np.zeros(MAX_HISTORY, dtype=np.int64)
        self._state_ep = np.full(MAX_HISTORY, -1, dtype=np.int64)
        self._state_halfmove = np.zeros(MAX_HISTORY, dtype=np.int64)
        self._state_fullmove = np.ones(MAX_HISTORY, dtype=np.int64)
        self._state_moves = [None] * MAX_HISTORY
        self._ply = 0
        self._move_stack = []
        self._hash_history = []
        self.reset()
        if fen:
            self.set_fen(fen)

    def reset(self):
        self._pieces[:] = 0
        self._pieces[0] = np.uint64(0x000000000000FF00)
        self._pieces[1] = np.uint64(0x0000000000000042)
        self._pieces[2] = np.uint64(0x0000000000000024)
        self._pieces[3] = np.uint64(0x0000000000000081)
        self._pieces[4] = np.uint64(0x0000000000000008)
        self._pieces[5] = np.uint64(0x0000000000000010)
        self._pieces[6] = np.uint64(0x00FF000000000000)
        self._pieces[7] = np.uint64(0x4200000000000000)
        self._pieces[8] = np.uint64(0x2400000000000000)
        self._pieces[9] = np.uint64(0x8100000000000000)
        self._pieces[10] = np.uint64(0x0800000000000000)
        self._pieces[11] = np.uint64(0x1000000000000000)
        self.turn = WHITE
        self.castling_rights = 15
        self.ep_square = -1
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._ply = 0
        self._move_stack.clear()
        self._hash_history = [polyglot.zobrist_hash(self)]

    @property
    def move_stack(self):
        return self._move_stack

    def pieces(self, piece_type, color):
        return int(self._pieces[(piece_type - 1) + (0 if color else 6)])

    def piece_at(self, square):
        bit = 1 << square
        for index in range(12):
            if int(self._pieces[index]) & bit:
                return Piece(index % 6 + 1, index < 6)
        return None

    def piece_type_at(self, square):
        piece = self.piece_at(square)
        return piece.piece_type if piece else None

    def piece_count(self):
        return sum(int(bitboard).bit_count() for bitboard in self._pieces)

    def _king_square(self, color):
        bits = self._pieces[(KING - 1) + (0 if color else 6)]
        return (int(bits) & -int(bits)).bit_length() - 1

    def is_check(self):
        return bool(
            is_attacked(self._king_square(self.turn), not self.turn, self._pieces)
        )

    def see(self, move):
        return int(see(self._pieces, self.turn, move.from_square, move.to_square))

    def fast_see(self, move, threshold=0):
        return bool(
            fast_see(
                self._pieces,
                self.turn,
                move.from_square,
                move.to_square,
                threshold,
            )
        )

    @property
    def legal_moves(self):
        pseudo = np.empty(256, dtype=np.int64)
        legal = np.empty(256, dtype=np.int64)
        _, count = _generate_legal_into(
            self._pieces,
            self.turn,
            self.castling_rights,
            self.ep_square,
            pseudo,
            legal,
        )
        return [Move.from_value(int(legal[index])) for index in range(count)]

    def generate_legal_captures(self):
        return [move for move in self.legal_moves if self.is_capture(move)]

    def is_capture(self, move):
        return self.piece_at(move.to_square) is not None or self.is_en_passant(move)

    def is_en_passant(self, move):
        return (
            move.to_square == self.ep_square
            and self.piece_type_at(move.from_square) == PAWN
            and self.piece_at(move.to_square) is None
        )

    def is_castling(self, move):
        return (
            self.piece_type_at(move.from_square) == KING
            and abs(move.to_square - move.from_square) == 2
        )

    def push(self, move):
        self._state_pieces[self._ply] = self._pieces
        self._state_turn[self._ply] = self.turn
        self._state_castling[self._ply] = self.castling_rights
        self._state_ep[self._ply] = self.ep_square
        self._state_halfmove[self._ply] = self.halfmove_clock
        self._state_fullmove[self._ply] = self.fullmove_number
        self._state_moves[self._ply] = move
        self._ply += 1
        self._move_stack.append(move)

        if move == Move.null():
            self.turn = not self.turn
            self.ep_square = -1
            self._hash_history.append(polyglot.zobrist_hash(self))
            return

        moving = self.piece_at(move.from_square)
        if moving is None:
            raise ValueError("move has no piece")
        is_ep = self.is_en_passant(move)
        is_castle = self.is_castling(move)
        source_bit = np.uint64(1) << np.uint64(move.from_square)
        target_bit = np.uint64(1) << np.uint64(move.to_square)
        moving_index = (moving.piece_type - 1) + (0 if moving.color else 6)
        self._pieces[moving_index] &= ~source_bit
        captured = self.piece_at(move.to_square)
        if captured:
            captured_index = (captured.piece_type - 1) + (0 if captured.color else 6)
            self._pieces[captured_index] &= ~target_bit
        if is_ep:
            capture_square = move.to_square - 8 if moving.color else move.to_square + 8
            self._pieces[(PAWN - 1) + (0 if not moving.color else 6)] &= ~(
                np.uint64(1) << np.uint64(capture_square)
            )
        placed_type = move.promotion or moving.piece_type
        placed_index = (placed_type - 1) + (0 if moving.color else 6)
        self._pieces[placed_index] |= target_bit
        if is_castle:
            rook_from, rook_to = (
                (H1, F1)
                if move.to_square == G1
                else (A1, D1)
                if move.to_square == C1
                else (H8, F8)
                if move.to_square == G8
                else (A8, D8)
            )
            rook_index = (ROOK - 1) + (0 if moving.color else 6)
            self._pieces[rook_index] &= ~(np.uint64(1) << np.uint64(rook_from))
            self._pieces[rook_index] |= np.uint64(1) << np.uint64(rook_to)
        if moving.piece_type == KING:
            self.castling_rights &= ~(3 if moving.color else 12)
        if moving.piece_type == ROOK:
            self.castling_rights &= ~(
                2
                if move.from_square == A1
                else 1
                if move.from_square == H1
                else 8
                if move.from_square == A8
                else 4
                if move.from_square == H8
                else 0
            )
        if captured and captured.piece_type == ROOK:
            self.castling_rights &= ~(
                2
                if move.to_square == A1
                else 1
                if move.to_square == H1
                else 8
                if move.to_square == A8
                else 4
                if move.to_square == H8
                else 0
            )
        self.ep_square = (
            (move.from_square + move.to_square) // 2
            if moving.piece_type == PAWN
            and abs(move.to_square - move.from_square) == 16
            else -1
        )
        self.halfmove_clock = (
            0 if moving.piece_type == PAWN or captured else self.halfmove_clock + 1
        )
        if not moving.color:
            self.fullmove_number += 1
        self.turn = not self.turn
        self._hash_history.append(polyglot.zobrist_hash(self))

    def pop(self):
        self._ply -= 1
        move = self._state_moves[self._ply]
        self._pieces[:] = self._state_pieces[self._ply]
        self.turn = bool(self._state_turn[self._ply])
        self.castling_rights = int(self._state_castling[self._ply])
        self.ep_square = int(self._state_ep[self._ply])
        self.halfmove_clock = int(self._state_halfmove[self._ply])
        self.fullmove_number = int(self._state_fullmove[self._ply])
        self._move_stack.pop()
        self._hash_history.pop()
        return move

    def is_legal(self, move):
        return move in self.legal_moves

    def is_repetition(self, count):
        current = self._hash_history[-1]
        return self._hash_history.count(current) >= count

    def can_claim_fifty_moves(self):
        return self.halfmove_clock >= 100

    def copy(self):
        result = Board()
        result._pieces = self._pieces.copy()
        result.turn = self.turn
        result.castling_rights = self.castling_rights
        result.ep_square = self.ep_square
        result.halfmove_clock = self.halfmove_clock
        result.fullmove_number = self.fullmove_number
        result._hash_history = self._hash_history.copy()
        return result

    def set_fen(self, fen):
        fields = fen.split()
        self._pieces[:] = 0
        rank = 7
        file = 0
        symbols = {
            "P": (PAWN, WHITE),
            "N": (KNIGHT, WHITE),
            "B": (BISHOP, WHITE),
            "R": (ROOK, WHITE),
            "Q": (QUEEN, WHITE),
            "K": (KING, WHITE),
            "p": (PAWN, BLACK),
            "n": (KNIGHT, BLACK),
            "b": (BISHOP, BLACK),
            "r": (ROOK, BLACK),
            "q": (QUEEN, BLACK),
            "k": (KING, BLACK),
        }
        for char in fields[0]:
            if char == "/":
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                piece_type, color = symbols[char]
                index = piece_type - 1 + (0 if color else 6)
                self._pieces[index] |= np.uint64(1) << np.uint64(rank * 8 + file)
                file += 1
        self.turn = fields[1] == "w"
        self.castling_rights = sum(
            value
            for token, value in (("K", 1), ("Q", 2), ("k", 4), ("q", 8))
            if token in fields[2]
        )
        self.ep_square = (
            -1
            if fields[3] == "-"
            else (ord(fields[3][0]) - 97) + (ord(fields[3][1]) - 49) * 8
        )
        self.halfmove_clock = int(fields[4])
        self.fullmove_number = int(fields[5])
        self._ply = 0
        self._move_stack.clear()
        self._hash_history = [polyglot.zobrist_hash(self)]

    def perft(self, depth):
        return int(
            _perft_encoded(
                self._pieces,
                self.turn,
                self.castling_rights,
                self.ep_square,
                depth,
            )
        )
