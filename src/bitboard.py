import os
import struct

import numpy as np
from numba import njit

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

    bishops_queens = (
        pieces[2] | pieces[8] | pieces[4] | pieces[10]
    )
    rooks_queens = (
        pieces[3] | pieces[9] | pieces[4] | pieces[10]
    )

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
        w_pawns_att
        | b_pawns_att
        | knights_att
        | kings_att
        | bishops_att
        | rooks_att
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
        pieces[own_start + ROOK - 1] |= np.uint64(1) << np.uint64(
            rook_to + rank_offset
        )

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


# fmt: off
_POLYGLOT_KEYS = np.array(
    [
        0x9D39247E33776D41, 0x2AF7398005AAA5C7, 0x44DB015024623547, 0x9C15F73E62A76AE2,
        0x75834465489C0C89, 0x3290AC3A203001BF, 0x0FBBAD1F61042279, 0xE83A908FF2FB60CA,
        0x0D7E765D58755C10, 0x1A083822CEAFE02D, 0x9605D5F0E25EC3B0, 0xD021FF5CD13A2ED5,
        0x40BDF15D4A672E32, 0x011355146FD56395, 0x5DB4832046F3D9E5, 0x239F8B2D7FF719CC,
        0x05D1A1AE85B49AA1, 0x679F848F6E8FC971, 0x7449BBFF801FED0B, 0x7D11CDB1C3B7ADF0,
        0x82C7709E781EB7CC, 0xF3218F1C9510786C, 0x331478F3AF51BBE6, 0x4BB38DE5E7219443,
        0xAA649C6EBCFD50FC, 0x8DBD98A352AFD40B, 0x87D2074B81D79217, 0x19F3C751D3E92AE1,
        0xB4AB30F062B19ABF, 0x7B0500AC42047AC4, 0xC9452CA81A09D85D, 0x24AA6C514DA27500,
        0x4C9F34427501B447, 0x14A68FD73C910841, 0xA71B9B83461CBD93, 0x03488B95B0F1850F,
        0x637B2B34FF93C040, 0x09D1BC9A3DD90A94, 0x3575668334A1DD3B, 0x735E2B97A4C45A23,
        0x18727070F1BD400B, 0x1FCBACD259BF02E7, 0xD310A7C2CE9B6555, 0xBF983FE0FE5D8244,
        0x9F74D14F7454A824, 0x51EBDC4AB9BA3035, 0x5C82C505DB9AB0FA, 0xFCF7FE8A3430B241,
        0x3253A729B9BA3DDE, 0x8C74C368081B3075, 0xB9BC6C87167C33E7, 0x7EF48F2B83024E20,
        0x11D505D4C351BD7F, 0x6568FCA92C76A243, 0x4DE0B0F40F32A7B8, 0x96D693460CC37E5D,
        0x42E240CB63689F2F, 0x6D2BDCDAE2919661, 0x42880B0236E4D951, 0x5F0F4A5898171BB6,
        0x39F890F579F92F88, 0x93C5B5F47356388B, 0x63DC359D8D231B78, 0xEC16CA8AEA98AD76,
        0x5355F900C2A82DC7, 0x07FB9F855A997142, 0x5093417AA8A7ED5E, 0x7BCBC38DA25A7F3C,
        0x19FC8A768CF4B6D4, 0x637A7780DECFC0D9, 0x8249A47AEE0E41F7, 0x79AD695501E7D1E8,
        0x14ACBAF4777D5776, 0xF145B6BECCDEA195, 0xDABF2AC8201752FC, 0x24C3C94DF9C8D3F6,
        0xBB6E2924F03912EA, 0x0CE26C0B95C980D9, 0xA49CD132BFBF7CC4, 0xE99D662AF4243939,
        0x27E6AD7891165C3F, 0x8535F040B9744FF1, 0x54B3F4FA5F40D873, 0x72B12C32127FED2B,
        0xEE954D3C7B411F47, 0x9A85AC909A24EAA1, 0x70AC4CD9F04F21F5, 0xF9B89D3E99A075C2,
        0x87B3E2B2B5C907B1, 0xA366E5B8C54F48B8, 0xAE4A9346CC3F7CF2, 0x1920C04D47267BBD,
        0x87BF02C6B49E2AE9, 0x092237AC237F3859, 0xFF07F64EF8ED14D0, 0x8DE8DCA9F03CC54E,
        0x9C1633264DB49C89, 0xB3F22C3D0B0B38ED, 0x390E5FB44D01144B, 0x5BFEA5B4712768E9,
        0x1E1032911FA78984, 0x9A74ACB964E78CB3, 0x4F80F7A035DAFB04, 0x6304D09A0B3738C4,
        0x2171E64683023A08, 0x5B9B63EB9CEFF80C, 0x506AACF489889342, 0x1881AFC9A3A701D6,
        0x6503080440750644, 0xDFD395339CDBF4A7, 0xEF927DBCF00C20F2, 0x7B32F7D1E03680EC,
        0xB9FD7620E7316243, 0x05A7E8A57DB91B77, 0xB5889C6E15630A75, 0x4A750A09CE9573F7,
        0xCF464CEC899A2F8A, 0xF538639CE705B824, 0x3C79A0FF5580EF7F, 0xEDE6C87F8477609D,
        0x799E81F05BC93F31, 0x86536B8CF3428A8C, 0x97D7374C60087B73, 0xA246637CFF328532,
        0x043FCAE60CC0EBA0, 0x920E449535DD359E, 0x70EB093B15B290CC, 0x73A1921916591CBD,
        0x56436C9FE1A1AA8D, 0xEFAC4B70633B8F81, 0xBB215798D45DF7AF, 0x45F20042F24F1768,
        0x930F80F4E8EB7462, 0xFF6712FFCFD75EA1, 0xAE623FD67468AA70, 0xDD2C5BC84BC8D8FC,
        0x7EED120D54CF2DD9, 0x22FE545401165F1C, 0xC91800E98FB99929, 0x808BD68E6AC10365,
        0xDEC468145B7605F6, 0x1BEDE3A3AEF53302, 0x43539603D6C55602, 0xAA969B5C691CCB7A,
        0xA87832D392EFEE56, 0x65942C7B3C7E11AE, 0xDED2D633CAD004F6, 0x21F08570F420E565,
        0xB415938D7DA94E3C, 0x91B859E59ECB6350, 0x10CFF333E0ED804A, 0x28AED140BE0BB7DD,
        0xC5CC1D89724FA456, 0x5648F680F11A2741, 0x2D255069F0B7DAB3, 0x9BC5A38EF729ABD4,
        0xEF2F054308F6A2BC, 0xAF2042F5CC5C2858, 0x480412BAB7F5BE2A, 0xAEF3AF4A563DFE43,
        0x19AFE59AE451497F, 0x52593803DFF1E840, 0xF4F076E65F2CE6F0, 0x11379625747D5AF3,
        0xBCE5D2248682C115, 0x9DA4243DE836994F, 0x066F70B33FE09017, 0x4DC4DE189B671A1C,
        0x51039AB7712457C3, 0xC07A3F80C31FB4B4, 0xB46EE9C5E64A6E7C, 0xB3819A42ABE61C87,
        0x21A007933A522A20, 0x2DF16F761598AA4F, 0x763C4A1371B368FD, 0xF793C46702E086A0,
        0xD7288E012AEB8D31, 0xDE336A2A4BC1C44B, 0x0BF692B38D079F23, 0x2C604A7A177326B3,
        0x4850E73E03EB6064, 0xCFC447F1E53C8E1B, 0xB05CA3F564268D99, 0x9AE182C8BC9474E8,
        0xA4FC4BD4FC5558CA, 0xE755178D58FC4E76, 0x69B97DB1A4C03DFE, 0xF9B5B7C4ACC67C96,
        0xFC6A82D64B8655FB, 0x9C684CB6C4D24417, 0x8EC97D2917456ED0, 0x6703DF9D2924E97E,
        0xC547F57E42A7444E, 0x78E37644E7CAD29E, 0xFE9A44E9362F05FA, 0x08BD35CC38336615,
        0x9315E5EB3A129ACE, 0x94061B871E04DF75, 0xDF1D9F9D784BA010, 0x3BBA57B68871B59D,
        0xD2B7ADEEDED1F73F, 0xF7A255D83BC373F8, 0xD7F4F2448C0CEB81, 0xD95BE88CD210FFA7,
        0x336F52F8FF4728E7, 0xA74049DAC312AC71, 0xA2F61BB6E437FDB5, 0x4F2A5CB07F6A35B3,
        0x87D380BDA5BF7859, 0x16B9F7E06C453A21, 0x7BA2484C8A0FD54E, 0xF3A678CAD9A2E38C,
        0x39B0BF7DDE437BA2, 0xFCAF55C1BF8A4424, 0x18FCF680573FA594, 0x4C0563B89F495AC3,
        0x40E087931A00930D, 0x8CFFA9412EB642C1, 0x68CA39053261169F, 0x7A1EE967D27579E2,
        0x9D1D60E5076F5B6F, 0x3810E399B6F65BA2, 0x32095B6D4AB5F9B1, 0x35CAB62109DD038A,
        0xA90B24499FCFAFB1, 0x77A225A07CC2C6BD, 0x513E5E634C70E331, 0x4361C0CA3F692F12,
        0xD941ACA44B20A45B, 0x528F7C8602C5807B, 0x52AB92BEB9613989, 0x9D1DFA2EFC557F73,
        0x722FF175F572C348, 0x1D1260A51107FE97, 0x7A249A57EC0C9BA2, 0x04208FE9E8F7F2D6,
        0x5A110C6058B920A0, 0x0CD9A497658A5698, 0x56FD23C8F9715A4C, 0x284C847B9D887AAE,
        0x04FEABFBBDB619CB, 0x742E1E651C60BA83, 0x9A9632E65904AD3C, 0x881B82A13B51B9E2,
        0x506E6744CD974924, 0xB0183DB56FFC6A79, 0x0ED9B915C66ED37E, 0x5E11E86D5873D484,
        0xF678647E3519AC6E, 0x1B85D488D0F20CC5, 0xDAB9FE6525D89021, 0x0D151D86ADB73615,
        0xA865A54EDCC0F019, 0x93C42566AEF98FFB, 0x99E7AFEABE000731, 0x48CBFF086DDF285A,
        0x7F9B6AF1EBF78BAF, 0x58627E1A149BBA21, 0x2CD16E2ABD791E33, 0xD363EFF5F0977996,
        0x0CE2A38C344A6EED, 0x1A804AADB9CFA741, 0x907F30421D78C5DE, 0x501F65EDB3034D07,
        0x37624AE5A48FA6E9, 0x957BAF61700CFF4E, 0x3A6C27934E31188A, 0xD49503536ABCA345,
        0x088E049589C432E0, 0xF943AEE7FEBF21B8, 0x6C3B8E3E336139D3, 0x364F6FFA464EE52E,
        0xD60F6DCEDC314222, 0x56963B0DCA418FC0, 0x16F50EDF91E513AF, 0xEF1955914B609F93,
        0x565601C0364E3228, 0xECB53939887E8175, 0xBAC7A9A18531294B, 0xB344C470397BBA52,
        0x65D34954DAF3CEBD, 0xB4B81B3FA97511E2, 0xB422061193D6F6A7, 0x071582401C38434D,
        0x7A13F18BBEDC4FF5, 0xBC4097B116C524D2, 0x59B97885E2F2EA28, 0x99170A5DC3115544,
        0x6F423357E7C6A9F9, 0x325928EE6E6F8794, 0xD0E4366228B03343, 0x565C31F7DE89EA27,
        0x30F5611484119414, 0xD873DB391292ED4F, 0x7BD94E1D8E17DEBC, 0xC7D9F16864A76E94,
        0x947AE053EE56E63C, 0xC8C93882F9475F5F, 0x3A9BF55BA91F81CA, 0xD9A11FBB3D9808E4,
        0x0FD22063EDC29FCA, 0xB3F256D8ACA0B0B9, 0xB03031A8B4516E84, 0x35DD37D5871448AF,
        0xE9F6082B05542E4E, 0xEBFAFA33D7254B59, 0x9255ABB50D532280, 0xB9AB4CE57F2D34F3,
        0x693501D628297551, 0xC62C58F97DD949BF, 0xCD454F8F19C5126A, 0xBBE83F4ECC2BDECB,
        0xDC842B7E2819E230, 0xBA89142E007503B8, 0xA3BC941D0A5061CB, 0xE9F6760E32CD8021,
        0x09C7E552BC76492F, 0x852F54934DA55CC9, 0x8107FCCF064FCF56, 0x098954D51FFF6580,
        0x23B70EDB1955C4BF, 0xC330DE426430F69D, 0x4715ED43E8A45C0A, 0xA8D7E4DAB780A08D,
        0x0572B974F03CE0BB, 0xB57D2E985E1419C7, 0xE8D9ECBE2CF3D73F, 0x2FE4B17170E59750,
        0x11317BA87905E790, 0x7FBF21EC8A1F45EC, 0x1725CABFCB045B00, 0x964E915CD5E2B207,
        0x3E2B8BCBF016D66D, 0xBE7444E39328A0AC, 0xF85B2B4FBCDE44B7, 0x49353FEA39BA63B1,
        0x1DD01AAFCD53486A, 0x1FCA8A92FD719F85, 0xFC7C95D827357AFA, 0x18A6A990C8B35EBD,
        0xCCCB7005C6B9C28D, 0x3BDBB92C43B17F26, 0xAA70B5B4F89695A2, 0xE94C39A54A98307F,
        0xB7A0B174CFF6F36E, 0xD4DBA84729AF48AD, 0x2E18BC1AD9704A68, 0x2DE0966DAF2F8B1C,
        0xB9C11D5B1E43A07E, 0x64972D68DEE33360, 0x94628D38D0C20584, 0xDBC0D2B6AB90A559,
        0xD2733C4335C6A72F, 0x7E75D99D94A70F4D, 0x6CED1983376FA72B, 0x97FCAACBF030BC24,
        0x7B77497B32503B12, 0x8547EDDFB81CCB94, 0x79999CDFF70902CB, 0xCFFE1939438E9B24,
        0x829626E3892D95D7, 0x92FAE24291F2B3F1, 0x63E22C147B9C3403, 0xC678B6D860284A1C,
        0x5873888850659AE7, 0x0981DCD296A8736D, 0x9F65789A6509A440, 0x9FF38FED72E9052F,
        0xE479EE5B9930578C, 0xE7F28ECD2D49EECD, 0x56C074A581EA17FE, 0x5544F7D774B14AEF,
        0x7B3F0195FC6F290F, 0x12153635B2C0CF57, 0x7F5126DBBA5E0CA7, 0x7A76956C3EAFB413,
        0x3D5774A11D31AB39, 0x8A1B083821F40CB4, 0x7B4A38E32537DF62, 0x950113646D1D6E03,
        0x4DA8979A0041E8A9, 0x3BC36E078F7515D7, 0x5D0A12F27AD310D1, 0x7F9D1A2E1EBE1327,
        0xDA3A361B1C5157B1, 0xDCDD7D20903D0C25, 0x36833336D068F707, 0xCE68341F79893389,
        0xAB9090168DD05F34, 0x43954B3252DC25E5, 0xB438C2B67F98E5E9, 0x10DCD78E3851A492,
        0xDBC27AB5447822BF, 0x9B3CDB65F82CA382, 0xB67B7896167B4C84, 0xBFCED1B0048EAC50,
        0xA9119B60369FFEBD, 0x1FFF7AC80904BF45, 0xAC12FB171817EEE7, 0xAF08DA9177DDA93D,
        0x1B0CAB936E65C744, 0xB559EB1D04E5E932, 0xC37B45B3F8D6F2BA, 0xC3A9DC228CAAC9E9,
        0xF3B8B6675A6507FF, 0x9FC477DE4ED681DA, 0x67378D8ECCEF96CB, 0x6DD856D94D259236,
        0xA319CE15B0B4DB31, 0x073973751F12DD5E, 0x8A8E849EB32781A5, 0xE1925C71285279F5,
        0x74C04BF1790C0EFE, 0x4DDA48153C94938A, 0x9D266D6A1CC0542C, 0x7440FB816508C4FE,
        0x13328503DF48229F, 0xD6BF7BAEE43CAC40, 0x4838D65F6EF6748F, 0x1E152328F3318DEA,
        0x8F8419A348F296BF, 0x72C8834A5957B511, 0xD7A023A73260B45C, 0x94EBC8ABCFB56DAE,
        0x9FC10D0F989993E0, 0xDE68A2355B93CAE6, 0xA44CFE79AE538BBE, 0x9D1D84FCCE371425,
        0x51D2B1AB2DDFB636, 0x2FD7E4B9E72CD38C, 0x65CA5B96B7552210, 0xDD69A0D8AB3B546D,
        0x604D51B25FBF70E2, 0x73AA8A564FB7AC9E, 0x1A8C1E992B941148, 0xAAC40A2703D9BEA0,
        0x764DBEAE7FA4F3A6, 0x1E99B96E70A9BE8B, 0x2C5E9DEB57EF4743, 0x3A938FEE32D29981,
        0x26E6DB8FFDF5ADFE, 0x469356C504EC9F9D, 0xC8763C5B08D1908C, 0x3F6C6AF859D80055,
        0x7F7CC39420A3A545, 0x9BFB227EBDF4C5CE, 0x89039D79D6FC5C5C, 0x8FE88B57305E2AB6,
        0xA09E8C8C35AB96DE, 0xFA7E393983325753, 0xD6B6D0ECC617C699, 0xDFEA21EA9E7557E3,
        0xB67C1FA481680AF8, 0xCA1E3785A9E724E5, 0x1CFC8BED0D681639, 0xD18D8549D140CAEA,
        0x4ED0FE7E9DC91335, 0xE4DBF0634473F5D2, 0x1761F93A44D5AEFE, 0x53898E4C3910DA55,
        0x734DE8181F6EC39A, 0x2680B122BAA28D97, 0x298AF231C85BAFAB, 0x7983EED3740847D5,
        0x66C1A2A1A60CD889, 0x9E17E49642A3E4C1, 0xEDB454E7BADC0805, 0x50B704CAB602C329,
        0x4CC317FB9CDDD023, 0x66B4835D9EAFEA22, 0x219B97E26FFC81BD, 0x261E4E4C0A333A9D,
        0x1FE2CCA76517DB90, 0xD7504DFA8816EDBB, 0xB9571FA04DC089C8, 0x1DDC0325259B27DE,
        0xCF3F4688801EB9AA, 0xF4F5D05C10CAB243, 0x38B6525C21A42B0E, 0x36F60E2BA4FA6800,
        0xEB3593803173E0CE, 0x9C4CD6257C5A3603, 0xAF0C317D32ADAA8A, 0x258E5A80C7204C4B,
        0x8B889D624D44885D, 0xF4D14597E660F855, 0xD4347F66EC8941C3, 0xE699ED85B0DFB40D,
        0x2472F6207C2D0484, 0xC2A1E7B5B459AEB5, 0xAB4F6451CC1D45EC, 0x63767572AE3D6174,
        0xA59E0BD101731A28, 0x116D0016CB948F09, 0x2CF9C8CA052F6E9F, 0x0B090A7560A968E3,
        0xABEEDDB2DDE06FF1, 0x58EFC10B06A2068D, 0xC6E57A78FBD986E0, 0x2EAB8CA63CE802D7,
        0x14A195640116F336, 0x7C0828DD624EC390, 0xD74BBE77E6116AC7, 0x804456AF10F5FB53,
        0xEBE9EA2ADF4321C7, 0x03219A39EE587A30, 0x49787FEF17AF9924, 0xA1E9300CD8520548,
        0x5B45E522E4B1B4EF, 0xB49C3B3995091A36, 0xD4490AD526F14431, 0x12A8F216AF9418C2,
        0x001F837CC7350524, 0x1877B51E57A764D5, 0xA2853B80F17F58EE, 0x993E1DE72D36D310,
        0xB3598080CE64A656, 0x252F59CF0D9F04BB, 0xD23C8E176D113600, 0x1BDA0492E7E4586E,
        0x21E0BD5026C619BF, 0x3B097ADAF088F94E, 0x8D14DEDB30BE846E, 0xF95CFFA23AF5F6F4,
        0x3871700761B3F743, 0xCA672B91E9E4FA16, 0x64C8E531BFF53B55, 0x241260ED4AD1E87D,
        0x106C09B972D2E822, 0x7FBA195410E5CA30, 0x7884D9BC6CB569D8, 0x0647DFEDCD894A29,
        0x63573FF03E224774, 0x4FC8E9560F91B123, 0x1DB956E450275779, 0xB8D91274B9E9D4FB,
        0xA2EBEE47E2FBFCE1, 0xD9F1F30CCD97FB09, 0xEFED53D75FD64E6B, 0x2E6D02C36017F67F,
        0xA9AA4D20DB084E9B, 0xB64BE8D8B25396C1, 0x70CB6AF7C2D5BCF0, 0x98F076A4F7A2322E,
        0xBF84470805E69B5F, 0x94C3251F06F90CF3, 0x3E003E616A6591E9, 0xB925A6CD0421AFF3,
        0x61BDD1307C66E300, 0xBF8D5108E27E0D48, 0x240AB57A8B888B20, 0xFC87614BAF287E07,
        0xEF02CDD06FFDB432, 0xA1082C0466DF6C0A, 0x8215E577001332C8, 0xD39BB9C3A48DB6CF,
        0x2738259634305C14, 0x61CF4F94C97DF93D, 0x1B6BACA2AE4E125B, 0x758F450C88572E0B,
        0x959F587D507A8359, 0xB063E962E045F54D, 0x60E8ED72C0DFF5D1, 0x7B64978555326F9F,
        0xFD080D236DA814BA, 0x8C90FD9B083F4558, 0x106F72FE81E2C590, 0x7976033A39F7D952,
        0xA4EC0132764CA04B, 0x733EA705FAE4FA77, 0xB4D8F77BC3E56167, 0x9E21F4F903B33FD9,
        0x9D765E419FB69F6D, 0xD30C088BA61EA5EF, 0x5D94337FBFAF7F5B, 0x1A4E4822EB4D7A59,
        0x6FFE73E81B637FB3, 0xDDF957BC36D8B9CA, 0x64D0E29EEA8838B3, 0x08DD9BDFD96B9F63,
        0x087E79E5A57D1D13, 0xE328E230E3E2B3FB, 0x1C2559E30F0946BE, 0x720BF5F26F4D2EAA,
        0xB0774D261CC609DB, 0x443F64EC5A371195, 0x4112CF68649A260E, 0xD813F2FAB7F5C5CA,
        0x660D3257380841EE, 0x59AC2C7873F910A3, 0xE846963877671A17, 0x93B633ABFA3469F8,
        0xC0C0F5A60EF4CDCF, 0xCAF21ECD4377B28C, 0x57277707199B8175, 0x506C11B9D90E8B1D,
        0xD83CC2687A19255F, 0x4A29C6465A314CD1, 0xED2DF21216235097, 0xB5635C95FF7296E2,
        0x22AF003AB672E811, 0x52E762596BF68235, 0x9AEBA33AC6ECC6B0, 0x944F6DE09134DFB6,
        0x6C47BEC883A7DE39, 0x6AD047C430A12104, 0xA5B1CFDBA0AB4067, 0x7C45D833AFF07862,
        0x5092EF950A16DA0B, 0x9338E69C052B8E7B, 0x455A4B4CFE30E3F5, 0x6B02E63195AD0CF8,
        0x6B17B224BAD6BF27, 0xD1E0CCD25BB9C169, 0xDE0C89A556B9AE70, 0x50065E535A213CF6,
        0x9C1169FA2777B874, 0x78EDEFD694AF1EED, 0x6DC93D9526A50E68, 0xEE97F453F06791ED,
        0x32AB0EDB696703D3, 0x3A6853C7E70757A7, 0x31865CED6120F37D, 0x67FEF95D92607890,
        0x1F2B1D1F15F6DC9C, 0xB69E38A8965C6B65, 0xAA9119FF184CCCF4, 0xF43C732873F24C13,
        0xFB4A3D794A9A80D2, 0x3550C2321FD6109C, 0x371F77E76BB8417E, 0x6BFA9AAE5EC05779,
        0xCD04F3FF001A4778, 0xE3273522064480CA, 0x9F91508BFFCFC14A, 0x049A7F41061A9E60,
        0xFCB6BE43A9F2FE9B, 0x08DE8A1C7797DA9B, 0x8F9887E6078735A1, 0xB5B4071DBFC73A66,
        0x230E343DFBA08D33, 0x43ED7F5A0FAE657D, 0x3A88A0FBBCB05C63, 0x21874B8B4D2DBC4F,
        0x1BDEA12E35F6A8C9, 0x53C065C6C8E63528, 0xE34A1D250E7A8D6B, 0xD6B04D3B7651DD7E,
        0x5E90277E7CB39E2D, 0x2C046F22062DC67D, 0xB10BB459132D0A26, 0x3FA9DDFB67E2F199,
        0x0E09B88E1914F7AF, 0x10E8B35AF3EEAB37, 0x9EEDECA8E272B933, 0xD4C718BC4AE8AE5F,
        0x81536D601170FC20, 0x91B534F885818A06, 0xEC8177F83F900978, 0x190E714FADA5156E,
        0xB592BF39B0364963, 0x89C350C893AE7DC1, 0xAC042E70F8B383F2, 0xB49B52E587A1EE60,
        0xFB152FE3FF26DA89, 0x3E666E6F69AE2C15, 0x3B544EBE544C19F9, 0xE805A1E290CF2456,
        0x24B33C9D7ED25117, 0xE74733427B72F0C1, 0x0A804D18B7097475, 0x57E3306D881EDB4F,
        0x4AE7D6A36EB5DBCB, 0x2D8D5432157064C8, 0xD1E649DE1E7F268B, 0x8A328A1CEDFE552C,
        0x07A3AEC79624C7DA, 0x84547DDC3E203C94, 0x990A98FD5071D263, 0x1A4FF12616EEFC89,
        0xF6F7FD1431714200, 0x30C05B1BA332F41C, 0x8D2636B81555A786, 0x46C9FEB55D120902,
        0xCCEC0A73B49C9921, 0x4E9D2827355FC492, 0x19EBB029435DCB0F, 0x4659D2B743848A2C,
        0x963EF2C96B33BE31, 0x74F85198B05A2E7D, 0x5A0F544DD2B1FB18, 0x03727073C2E134B1,
        0xC7F6AA2DE59AEA61, 0x352787BAA0D7C22F, 0x9853EAB63B5E0B35, 0xABBDCDD7ED5C0860,
        0xCF05DAF5AC8D77B0, 0x49CAD48CEBF4A71E, 0x7A4C10EC2158C4A6, 0xD9E92AA246BF719E,
        0x13AE978D09FE5557, 0x730499AF921549FF, 0x4E4B705B92903BA4, 0xFF577222C14F0A3A,
        0x55B6344CF97AAFAE, 0xB862225B055B6960, 0xCAC09AFBDDD2CDB4, 0xDAF8E9829FE96B5F,
        0xB5FDFC5D3132C498, 0x310CB380DB6F7503, 0xE87FBB46217A360E, 0x2102AE466EBB1148,
        0xF8549E1A3AA5E00D, 0x07A69AFDCC42261A, 0xC4C118BFE78FEAAE, 0xF9F4892ED96BD438,
        0x1AF3DBE25D8F45DA, 0xF5B4B0B0D2DEEEB4, 0x962ACEEFA82E1C84, 0x046E3ECAAF453CE9,
        0xF05D129681949A4C, 0x964781CE734B3C84, 0x9C2ED44081CE5FBD, 0x522E23F3925E319E,
        0x177E00F9FC32F791, 0x2BC60A63A6F3B3F2, 0x222BBFAE61725606, 0x486289DDCC3D6780,
        0x7DC7785B8EFDFC80, 0x8AF38731C02BA980, 0x1FAB64EA29A2DDF7, 0xE4D9429322CD065A,
        0x9DA058C67844F20C, 0x24C0E332B70019B0, 0x233003B5A6CFE6AD, 0xD586BD01C5C217F6,
        0x5E5637885F29BC2B, 0x7EBA726D8C94094B, 0x0A56A5F0BFE39272, 0xD79476A84EE20D06,
        0x9E4C1269BAA4BF37, 0x17EFEE45B0DEE640, 0x1D95B0A5FCF90BC6, 0x93CBE0B699C2585D,
        0x65FA4F227A2B6D79, 0xD5F9E858292504D5, 0xC2B5A03F71471A6F, 0x59300222B4561E00,
        0xCE2F8642CA0712DC, 0x7CA9723FBB2E8988, 0x2785338347F2BA08, 0xC61BB3A141E50E8C,
        0x150F361DAB9DEC26, 0x9F6A419D382595F4, 0x64A53DC924FE7AC9, 0x142DE49FFF7A7C3D,
        0x0C335248857FA9E7, 0x0A9C32D5EAE45305, 0xE6C42178C4BBB92E, 0x71F1CE2490D20B07,
        0xF1BCC3D275AFE51A, 0xE728E8C83C334074, 0x96FBF83A12884624, 0x81A1549FD6573DA5,
        0x5FA7867CAF35E149, 0x56986E2EF3ED091B, 0x917F1DD5F8886C61, 0xD20D8C88C8FFE65F,
        0x31D71DCE64B2C310, 0xF165B587DF898190, 0xA57E6339DD2CF3A0, 0x1EF6E6DBB1961EC9,
        0x70CC73D90BC26E24, 0xE21A6B35DF0C3AD7, 0x003A93D8B2806962, 0x1C99DED33CB890A1,
        0xCF3145DE0ADD4289, 0xD0E4427A5514FB72, 0x77C621CC9FB3A483, 0x67A34DAC4356550B,
        0xF8D626AAAF278509
    ],
    dtype=np.uint64,
)
# fmt: on


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
            _POLYGLOT_KEYS,
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
            capture_square = (
                move.to_square - 8 if moving.color else move.to_square + 8
            )
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
