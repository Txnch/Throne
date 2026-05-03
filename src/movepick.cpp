#include "movepick.h"

#include "attacks.h"
#include "evaluate.h"
#include "movegen.h"
#include "bitboard.h"
#include "position.h"

#include <algorithm>

namespace {

static Bitboard all_attackers_to(const Position& pos, Square sq, Bitboard occ) {
    return attackers_to(pos, sq, occ, WHITE) | attackers_to(pos, sq, occ, BLACK);
}

static bool see_ge(const Position& pos, Move m, int threshold) {
    if (!is_capture(m) && !is_promotion(m))
        return true; 

    if (is_en_passant(m))
        return threshold <= 0; 

    Square from = from_sq(m);
    Square to = to_sq(m);


    int gain = 0;
    if (is_capture(m)) {
        gain = piece_value(piece_type(pos.piece_on(to)));
    }


    if (is_promotion(m)) {
        gain += piece_value(promotion_type(m)) - piece_value(PAWN);
    }


    if (gain < threshold)
        return false;


    int nextVal = is_promotion(m) ? piece_value(promotion_type(m)) : piece_value(piece_type(pos.piece_on(from)));

 
    if (gain - nextVal >= threshold)
        return true;


    Bitboard occ = pos.all_pieces() ^ square_bb(from) ^ square_bb(to);
    Color side = ~pos.side_to_move(); 


    Bitboard attackers = all_attackers_to(pos, to, occ);

    int balance = gain - nextVal - threshold;

    while (true) {
        Bitboard myAtt = attackers & pos.pieces(side);
        if (!myAtt)
            break; 

        PieceType pt = NO_PIECE_TYPE;
        int minV = 30000;

        for (int p = PAWN; p <= KING; ++p) {
            PieceType current_pt = static_cast<PieceType>(p);
            if (myAtt & pos.pieces(current_pt)) {
                minV = piece_value(current_pt);
                pt = current_pt;
                break; 
            }
        }


        Square attSq = lsb(myAtt & pos.pieces(pt));
        occ ^= square_bb(attSq);

        attackers |= (get_bishop_attacks(to, occ) & (pos.pieces(BISHOP) | pos.pieces(QUEEN)));
        attackers |= (get_rook_attacks(to, occ) & (pos.pieces(ROOK) | pos.pieces(QUEEN)));
        attackers &= occ; 


        balance = -balance - 1 - minV;
        nextVal = minV;
        side = ~side;

        if (balance >= 0) {
            if (pt == KING && (attackers & pos.pieces(side)))
                side = ~side;
            break;
        }
    }

    return side != pos.side_to_move();
}

static int capture_mvv_lva_internal(const Position& pos, Move m) {
    if (!is_capture(m))
        return 0;

    const Piece attacker = pos.piece_on(from_sq(m));

    Piece victim = pos.piece_on(to_sq(m));
    if (is_en_passant(m))
        victim = make_piece(~pos.side_to_move(), PAWN);

    return piece_value(piece_type(victim)) * 16 - piece_value(piece_type(attacker));
}

} // namespace

int movepick_capture_mvv_lva(const Position& pos, Move m) {
    return capture_mvv_lva_internal(pos, m);
}

bool movepick_see_ge(const Position& pos, Move m, int threshold) {
    return see_ge(pos, m, threshold);
}

MovePicker::MovePicker() = default;

int MovePicker::score_main_move(const Position& pos, Move m, const MovePicker::MainOrderData& order_data) {
    if (is_capture(m)) {
        int score = 200000 + movepick_capture_mvv_lva(pos, m);
        if (order_data.capture_history) {
            const Piece attacker = pos.piece_on(from_sq(m));
            Piece victim = pos.piece_on(to_sq(m));
            if (is_en_passant(m))
                victim = make_piece(~pos.side_to_move(), PAWN);
            if (attacker != NO_PIECE && victim != NO_PIECE)
                score += order_data.capture_history[attacker][victim] / 16;
        }
        return score;
    }

    int score = 0;
    if (order_data.history)
        score += order_data.history[from_sq(m)][to_sq(m)];

    const PieceType curPT = piece_type(pos.piece_on(from_sq(m)));
    if (order_data.cont1 && unsigned(curPT) < PIECE_TYPE_NB)
        score += order_data.cont1[curPT][to_sq(m)] / 4;
    if (order_data.cont2 && unsigned(curPT) < PIECE_TYPE_NB)
        score += order_data.cont2[curPT][to_sq(m)] / 4;
    if (order_data.quiet_check_bonus && pos.gives_check(m))
        score += order_data.quiet_check_bonus;
    return score;
}

void MovePicker::Bucket::reset() {
    count = 0;
    head = 0;
}

void MovePicker::Bucket::push(Move m, int s) {
    if (count >= MAX_MOVES)
        return;
    moves[count] = m;
    scores[count] = s;
    ++count;
}

bool MovePicker::Bucket::has_next() const {
    return head < count;
}

Move MovePicker::Bucket::pop_next(int& last_score_value, int fallback_score) {
    if (!has_next())
        return 0;

    int best = head;
    for (int i = head + 1; i < count; ++i)
        if (scores[i] > scores[best])
            best = i;

    if (best != head) {
        std::swap(moves[head], moves[best]);
        std::swap(scores[head], scores[best]);
    }

    last_score_value = scores[head];
    Move out = moves[head++];
    if (!out)
        last_score_value = fallback_score;
    return out;
}

void MovePicker::init_main(const Position& pos,
                           Move tt_move,
                           Move killer1,
                           Move killer2,
                           Move counter_move,
                           const MainOrderData* orderData) {
    stage = ST_DONE;

    pos_ptr = &pos;
    order_data = orderData ? *orderData : MainOrderData{};
    qs_in_check = false;

    tt = tt_move;
    killer_1 = killer1;
    killer_2 = killer2;
    counter = counter_move;

    has_tt = (tt != 0);
    has_k1 = (killer_1 != 0 && killer_1 != tt);
    has_k2 = (killer_2 != 0 && killer_2 != tt && killer_2 != killer_1);
    has_counter = (counter != 0 && counter != tt && counter != killer_1 && counter != killer_2);

    captures_generated = false;
    quiets_generated = false;
    qs_generated = false;

    captures.reset();
    bad_captures.reset();
    quiets.reset();
    qs_moves.reset();
    last_score_value = 0;

    stage = has_tt ? ST_TT : ST_GEN_CAPTURES;
}

void MovePicker::init_qsearch(const Position& pos,
                              bool in_check,
                              Move tt_move) {
    stage = ST_DONE;

    pos_ptr = &pos;
    order_data = MainOrderData{};
    qs_in_check = in_check;

    tt = tt_move;
    killer_1 = 0;
    killer_2 = 0;
    counter = 0;

    has_tt = (tt != 0);
    has_k1 = false;
    has_k2 = false;
    has_counter = false;

    captures_generated = false;
    quiets_generated = false;
    qs_generated = false;

    captures.reset();
    bad_captures.reset();
    quiets.reset();
    qs_moves.reset();
    last_score_value = 0;

    stage = has_tt ? ST_QS_TT : ST_QS_GEN;
}

Move MovePicker::next(bool skip_quiets) {
    if (!pos_ptr)
        return 0;

    for (;;) {
        switch (stage) {
        case ST_DONE:
            return 0;

        case ST_TT:
            stage = ST_GEN_CAPTURES;
            if (has_tt && pos_ptr->is_pseudo_legal(tt)) {
                last_score_value = 300000;
                return tt;
            }
            continue;

        case ST_GEN_CAPTURES: {
            if (!captures_generated) {
                MoveList list;
                generate_captures(*pos_ptr, list);

                for (int i = 0; i < list.size; ++i) {
                    const Move m = list.moves[i];
                    if (!m || m == tt)
                        continue;

                    int score = score_main_move(*pos_ptr, m, order_data);

                    if (is_capture(m)) {
                        const int see_margin = -movepick_capture_mvv_lva(*pos_ptr, m) / 32;
                        if (!see_ge(*pos_ptr, m, see_margin)) {
                            bad_captures.push(m, score);
                            continue;
                        }
                    }

                    captures.push(m, score);
                }

                captures_generated = true;
            }

            stage = ST_CAPTURES;
            continue;
        }

        case ST_CAPTURES: {
            Move m = captures.pop_next(last_score_value, 200000);
            if (m)
                return m;
            stage = ST_KILLER1;
            continue;
        }

        case ST_KILLER1:
            stage = ST_KILLER2;
            if (has_k1 && pos_ptr->is_pseudo_legal(killer_1) && !is_capture(killer_1) && !is_promotion(killer_1)) {
                last_score_value = 180000;
                return killer_1;
            }
            continue;

        case ST_KILLER2:
            stage = ST_COUNTER;
            if (has_k2 && pos_ptr->is_pseudo_legal(killer_2) && !is_capture(killer_2) && !is_promotion(killer_2)) {
                last_score_value = 170000;
                return killer_2;
            }
            continue;

        case ST_COUNTER:
            stage = ST_GEN_QUIETS;
            if (has_counter && pos_ptr->is_pseudo_legal(counter) && !is_capture(counter) && !is_promotion(counter)) {
                last_score_value = 160000;
                return counter;
            }
            continue;

        case ST_GEN_QUIETS: {
            if (skip_quiets) {
                stage = ST_BAD_CAPTURES;
                continue;
            }

            if (!quiets_generated) {
                MoveList list;
                generate_quiets(*pos_ptr, list);

                for (int i = 0; i < list.size; ++i) {
                    const Move m = list.moves[i];
                    if (!m)
                        continue;
                    if (m == tt || m == killer_1 || m == killer_2 || m == counter)
                        continue;
                    if (is_capture(m) || is_promotion(m))
                        continue;

                    const int score = score_main_move(*pos_ptr, m, order_data);
                    quiets.push(m, score);
                }

                quiets_generated = true;
            }

            stage = ST_QUIETS;
            continue;
        }

        case ST_QUIETS: {
            if (skip_quiets) {
                stage = ST_BAD_CAPTURES;
                continue;
            }

            Move m = quiets.pop_next(last_score_value, 0);
            if (m)
                return m;
            stage = ST_BAD_CAPTURES;
            continue;
        }

        case ST_BAD_CAPTURES: {
            Move m = bad_captures.pop_next(last_score_value, 100000);
            if (m)
                return m;
            stage = ST_DONE;
            continue;
        }

        case ST_QS_TT:
            stage = ST_QS_GEN;
            if (has_tt
                && pos_ptr->is_pseudo_legal(tt)
                && (qs_in_check || is_capture(tt) || is_promotion(tt))) {
                last_score_value = 300000;
                return tt;
            }
            continue;

        case ST_QS_GEN: {
            if (!qs_generated) {
                MoveList list;
                if (qs_in_check)
                    generate_moves(*pos_ptr, list);
                else
                    generate_captures(*pos_ptr, list);

                for (int i = 0; i < list.size; ++i) {
                    const Move m = list.moves[i];
                    if (!m || m == tt)
                        continue;

                    int score = is_capture(m) ? score_main_move(*pos_ptr, m, order_data) : 0;

                    qs_moves.push(m, score);
                }

                qs_generated = true;
            }

            stage = ST_QS_MOVES;
            continue;
        }

        case ST_QS_MOVES: {
            Move m = qs_moves.pop_next(last_score_value, 0);
            if (m)
                return m;
            stage = ST_DONE;
            continue;
        }
        }
    }
}







