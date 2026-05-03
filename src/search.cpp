#include "search.h"
#include "movegen.h"
#include "movepick.h"
#include "evaluate.h"
#include "attacks.h"
#include "move.h"
#include "timeman.h"
#include "tt.h"

#include <iostream>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <memory>

bool g_silent_search = false;

inline constexpr int INF = 30000;
inline constexpr int MATE_SCORE = 29000;

inline constexpr int HISTORY_DECAY_MASK = 63;
inline constexpr int HISTORY_CLAMP = 70000;


inline constexpr int MAIN_MOVE_QUIET_CHECK_BONUS = 8000;
inline constexpr int QUIET_HISTORY_COUNTERMOVE_BONUS = 12000;
inline constexpr int ROOT_PRIOR_BEST_BONUS = 25000;


inline constexpr int QS_MAX_CHECK_PLY = 32;
inline constexpr int QS_SEE_THRESHOLD = 0;


enum NodeType {
    Root,
    PV,
    NonPV
};


static int LMR_TABLE[2][64][256];


static int init_lmr_table = []() {
    for (int d = 1; d < 64; ++d) {
        for (int m = 1; m < 256; ++m) {
            double base = std::log(d) * std::log(m) / 2.25;
            // N
            LMR_TABLE[0][d][m] = static_cast<int>(0.40 + base);
            // Q
            LMR_TABLE[1][d][m] = static_cast<int>(0.80 + base);
        }
    }
    return 0;
    }();

static Move killer_moves[2][MAX_PLY];
static int  history_heuristic[64][64];

static Move countermove[64][64];

static int cont_history[PIECE_TYPE_NB][64][PIECE_TYPE_NB][64];

static int cont_history_2ply[PIECE_TYPE_NB][64][PIECE_TYPE_NB][64];
static int capture_history[PIECE_NB][PIECE_NB];

// --- Evaluation Correction Tables ---
constexpr int CORR_SIZE = 16384;
constexpr int CORR_MASK = CORR_SIZE - 1;

static int pawn_corr[2][CORR_SIZE];
static int non_pawn_corr[2][2][CORR_SIZE];
static int cont_corr[PIECE_NB][64];

static inline int draw_score()
{
    return 0;
}

static std::atomic<bool> stop_flag(false);
static uint64_t nodes_count = 0;
static uint64_t target_max_nodes = 0;
static int seldepth = 0;

static Move pv_table[MAX_PLY][MAX_PLY];
static int  pv_length[MAX_PLY];

static TimeManager tm;
static uint32_t search_generation = 0;

static void decay_history_tables_if_needed()
{
    ++search_generation;
    if ((search_generation & HISTORY_DECAY_MASK) != 0)
        return;

    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j)
            history_heuristic[i][j] /= 2;

    for (int p1 = 0; p1 < PIECE_TYPE_NB; ++p1)
        for (int s1 = 0; s1 < 64; ++s1)
            for (int p2 = 0; p2 < PIECE_TYPE_NB; ++p2)
                for (int s2 = 0; s2 < 64; ++s2)
                {
                    cont_history[p1][s1][p2][s2] /= 2;
                    cont_history_2ply[p1][s1][p2][s2] /= 2;
                }

    for (int a = 0; a < PIECE_NB; ++a)
        for (int v = 0; v < PIECE_NB; ++v)
            capture_history[a][v] /= 2;
}

struct SearchStack {
    Move current_move = 0;
    Piece moved_piece = NO_PIECE;
    int static_eval = -INF;
    int raw_eval = -INF;
    int seen_moves = 0;
    bool played_capture = false;
    int cutoff_count = 0;
    bool tt_pv = false;
    int hist_score = 0;
    Move excluded_move = 0;
    nnue::AccumulatorPair acc{};
    bool acc_valid = false;
};

static inline int pt_index(PieceType pt);

// --- Eval Correction Helpers ---
static int get_eval_correction(const Position& pos, const SearchStack* ss, int ply)
{
    int stm = pos.side_to_move();
    int score = 0;

    score += pawn_corr[stm][pos.pawn_key() & CORR_MASK];
    score += non_pawn_corr[stm][WHITE][pos.non_pawn_key(WHITE) & CORR_MASK];
    score += non_pawn_corr[stm][BLACK][pos.non_pawn_key(BLACK) & CORR_MASK];

    if (ply >= 1 && ss[ply - 1].current_move != 0) {
        Piece prev_piece = ss[ply - 1].moved_piece;
        Square prev_to = to_sq(ss[ply - 1].current_move);
        score += cont_corr[prev_piece][prev_to];
    }

    return score / 128;
}

static void update_eval_correction(const Position& pos, const SearchStack* ss, int ply, int depth, int diff)
{
    int stm = pos.side_to_move();

    int bonus = std::clamp((diff * depth * depth) / 2, -8192, 8192);

    auto update_weight = [](int& entry, int b) {
        entry += (b - entry) / 32;
        };

    update_weight(pawn_corr[stm][pos.pawn_key() & CORR_MASK], bonus);
    update_weight(non_pawn_corr[stm][WHITE][pos.non_pawn_key(WHITE) & CORR_MASK], bonus);
    update_weight(non_pawn_corr[stm][BLACK][pos.non_pawn_key(BLACK) & CORR_MASK], bonus);

    if (ply >= 1 && ss[ply - 1].current_move != 0) {
        Piece prev_piece = ss[ply - 1].moved_piece;
        Square prev_to = to_sq(ss[ply - 1].current_move);
        update_weight(cont_corr[prev_piece][prev_to], bonus);
    }
}

static void update_time()
{
    if (stop_flag) return;

    if (tm.should_stop(nodes_count))
        stop_flag = true;

    if (!stop_flag && tm.hard_stop())
        stop_flag = true;
}

void stop_search_now()
{
    stop_flag = true;
}

void clear_search_state_for_new_game()
{
    stop_flag = false;
    nodes_count = 0;
    seldepth = 0;
    search_generation = 0;

    std::fill(&killer_moves[0][0], &killer_moves[0][0] + 2 * MAX_PLY, Move(0));
    std::fill(&history_heuristic[0][0], &history_heuristic[0][0] + 64 * 64, 0);
    std::fill(&countermove[0][0], &countermove[0][0] + 64 * 64, Move(0));
    std::fill(&cont_history[0][0][0][0],
        &cont_history[0][0][0][0] + PIECE_TYPE_NB * 64 * PIECE_TYPE_NB * 64,
        0);
    std::fill(&cont_history_2ply[0][0][0][0],
        &cont_history_2ply[0][0][0][0] + PIECE_TYPE_NB * 64 * PIECE_TYPE_NB * 64,
        0);
    std::fill(&capture_history[0][0], &capture_history[0][0] + PIECE_NB * PIECE_NB, 0);
    std::fill(&pv_table[0][0], &pv_table[0][0] + MAX_PLY * MAX_PLY, Move(0));
    std::fill(pv_length, pv_length + MAX_PLY, 0);

    std::fill(&pawn_corr[0][0], &pawn_corr[0][0] + 2 * CORR_SIZE, 0);
    std::fill(&non_pawn_corr[0][0][0], &non_pawn_corr[0][0][0] + 4 * CORR_SIZE, 0);
    std::fill(&cont_corr[0][0], &cont_corr[0][0] + PIECE_NB * 64, 0);

    tm.reset();
}

static int total_pieces(const Position& pos)
{
    return popcount(pos.pieces(WHITE) | pos.pieces(BLACK));
}

static bool has_non_pawn_material(const Position& pos, Color c)
{
    return (pos.pieces(c) & ~(pos.pieces(PAWN) | pos.pieces(KING))) != 0;
}

static bool has_insufficient_material(const Position& pos)
{
    if (pos.pieces(PAWN) || pos.pieces(ROOK) || pos.pieces(QUEEN))
        return false;

    const int wn = popcount(pos.pieces(WHITE) & pos.pieces(KNIGHT));
    const int bn = popcount(pos.pieces(BLACK) & pos.pieces(KNIGHT));
    const int wb = popcount(pos.pieces(WHITE) & pos.pieces(BISHOP));
    const int bb = popcount(pos.pieces(BLACK) & pos.pieces(BISHOP));

    const int wMinor = wn + wb;
    const int bMinor = bn + bb;

    if (wMinor == 0 && bMinor == 0)
        return true;

    if ((wMinor == 1 && bMinor == 0) || (wMinor == 0 && bMinor == 1))
        return true;

    if (wMinor == 1 && bMinor == 1 && ((wb == 1 && bb == 1) || (wn == 1 && bn == 1)))
        return true;

    if ((wn == 2 && wb == 0 && bMinor == 0) || (bn == 2 && bb == 0 && wMinor == 0))
        return true;

    return false;
}

static inline int pt_index(PieceType pt)
{
    return int(pt);
}

static inline Piece captured_piece_for_move(const Position& pos, Move m)
{
    if (!is_capture(m))
        return NO_PIECE;

    if (is_en_passant(m))
        return make_piece(~pos.side_to_move(), PAWN);

    return pos.piece_on(to_sq(m));
}

static inline int capture_history_score(const Position& pos, Move m)
{
    if (!is_capture(m))
        return 0;

    const Piece attacker = pos.piece_on(from_sq(m));
    const Piece victim = captured_piece_for_move(pos, m);
    if (attacker == NO_PIECE || victim == NO_PIECE)
        return 0;

    return capture_history[attacker][victim];
}

static inline void update_capture_history(Piece attacker, Piece victim, int delta)
{
    if (attacker == NO_PIECE || victim == NO_PIECE)
        return;

    int& h = capture_history[attacker][victim];
    h += delta;
    if (h > HISTORY_CLAMP) h = HISTORY_CLAMP;
    else if (h < -HISTORY_CLAMP) h = -HISTORY_CLAMP;
}

static inline int get_cont1_score(const SearchStack* ss, int ply, PieceType curPT, Square curTo)
{
    if (ply < 1) return 0;

    const SearchStack& prev1 = ss[ply - 1];
    if (!prev1.current_move || prev1.moved_piece == NO_PIECE)
        return 0;

    int p1 = pt_index(piece_type(prev1.moved_piece));
    int p2 = pt_index(curPT);

    if (unsigned(p1) < PIECE_TYPE_NB && unsigned(p2) < PIECE_TYPE_NB)
        return cont_history[p1][to_sq(prev1.current_move)][p2][curTo];

    return 0;
}

static inline int get_cont2_score(const SearchStack* ss, int ply, PieceType curPT, Square curTo)
{
    if (ply < 2) return 0;

    const SearchStack& prev2 = ss[ply - 2];
    if (!prev2.current_move || prev2.moved_piece == NO_PIECE)
        return 0;

    int p1 = pt_index(piece_type(prev2.moved_piece));
    int p2 = pt_index(curPT);

    if (unsigned(p1) < PIECE_TYPE_NB && unsigned(p2) < PIECE_TYPE_NB)
        return cont_history_2ply[p1][to_sq(prev2.current_move)][p2][curTo];

    return 0;
}

static inline int quiet_countermove_bonus(const SearchStack* ss, int ply, Move m)
{
    if (!ss || ply < 1)
        return 0;

    Move prev = ss[ply - 1].current_move;
    if (prev && m == countermove[from_sq(prev)][to_sq(prev)])
        return QUIET_HISTORY_COUNTERMOVE_BONUS;

    return 0;
}

static inline int quiet_history_score(Position& pos, Move m, int ply, const SearchStack* ss)
{
    const Square to = to_sq(m);
    const PieceType curPT = piece_type(pos.piece_on(from_sq(m)));

    int score = history_heuristic[from_sq(m)][to];
    score += get_cont1_score(ss, ply, curPT, to) / 2;
    score += get_cont2_score(ss, ply, curPT, to) / 2;
    score += quiet_countermove_bonus(ss, ply, m);
    return score;
}

static inline void update_continuation_histories(const SearchStack* ss, int ply, Move m, Piece movedPiece, int delta)
{
    const PieceType curPT = piece_type(movedPiece);
    const Square curTo = to_sq(m);
    const int p2 = pt_index(curPT);

    auto clamp_hist = [](int& x)
        {
            if (x > HISTORY_CLAMP) x = HISTORY_CLAMP;
            else if (x < -HISTORY_CLAMP) x = -HISTORY_CLAMP;
        };

    if (ply >= 1)
    {
        const SearchStack& prev1 = ss[ply - 1];
        if (prev1.current_move && prev1.moved_piece != NO_PIECE)
        {
            int p1 = pt_index(piece_type(prev1.moved_piece));
            if (unsigned(p1) < PIECE_TYPE_NB && unsigned(p2) < PIECE_TYPE_NB)
            {
                int& ch = cont_history[p1][to_sq(prev1.current_move)][p2][curTo];
                ch += delta;
                clamp_hist(ch);
            }
        }
    }

    if (ply >= 2)
    {
        const SearchStack& prev2 = ss[ply - 2];
        if (prev2.current_move && prev2.moved_piece != NO_PIECE)
        {
            int p1 = pt_index(piece_type(prev2.moved_piece));
            if (unsigned(p1) < PIECE_TYPE_NB && unsigned(p2) < PIECE_TYPE_NB)
            {
                int& ch2 = cont_history_2ply[p1][to_sq(prev2.current_move)][p2][curTo];
                ch2 += delta;
                clamp_hist(ch2);
            }
        }
    }
}

template <NodeType NT>
static int negamax(Position& pos, int depth, int alpha, int beta, int ply, SearchStack* ss);

static int score_root_move(Position& pos, Move m, Move prior_best_move)
{
    MovePicker::MainOrderData rootOrderData{};
    rootOrderData.history = history_heuristic;
    rootOrderData.capture_history = capture_history;
    rootOrderData.quiet_check_bonus = MAIN_MOVE_QUIET_CHECK_BONUS;

    int score = MovePicker::score_main_move(pos, m, rootOrderData);

    if (prior_best_move && m == prior_best_move)
        score += ROOT_PRIOR_BEST_BONUS;

    return score;
}

static MovePicker::MainOrderData build_main_order_data(const SearchStack* ss, int ply)
{
    MovePicker::MainOrderData data{};
    data.history = history_heuristic;
    data.capture_history = capture_history;
    data.quiet_check_bonus = MAIN_MOVE_QUIET_CHECK_BONUS;

    if (!ss || ply < 1)
        return data;

    const SearchStack& prev1 = ss[ply - 1];
    if (prev1.current_move && prev1.moved_piece != NO_PIECE)
    {
        const int p1 = pt_index(piece_type(prev1.moved_piece));
        if (unsigned(p1) < PIECE_TYPE_NB)
            data.cont1 = cont_history[p1][to_sq(prev1.current_move)];
    }

    if (ply < 2)
        return data;

    const SearchStack& prev2 = ss[ply - 2];
    if (prev2.current_move && prev2.moved_piece != NO_PIECE)
    {
        const int p2 = pt_index(piece_type(prev2.moved_piece));
        if (unsigned(p2) < PIECE_TYPE_NB)
            data.cont2 = cont_history_2ply[p2][to_sq(prev2.current_move)];
    }

    return data;
}

static inline void ensure_accumulator(const Position& pos, SearchStack* ss, int ply)
{
    if (!nnue::is_ready() || ss[ply].acc_valid)
        return;

    int valid_ply = ply - 1;
    while (valid_ply >= 0 && !ss[valid_ply].acc_valid)
        --valid_ply;

    if (valid_ply < 0) {
        nnue::refresh_acc(pos, WHITE, ss[ply].acc.white);
        nnue::refresh_acc(pos, BLACK, ss[ply].acc.black);
        ss[ply].acc_valid = true;
        return;
    }

    for (int i = valid_ply; i < ply; ++i) {
        ss[i + 1].acc = ss[i].acc;

        int hist_idx = pos.current_ply() - (ply - i);
        const nnue::DirtyPieces& dp = pos.state_at_ply(hist_idx).dp;

        nnue::apply_dirty(ss[i + 1].acc.white, WHITE, dp);
        nnue::apply_dirty(ss[i + 1].acc.black, BLACK, dp);
        ss[i + 1].acc_valid = true;
    }
}

static inline int eval_from_stack(const Position& pos, SearchStack* ss, int ply)
{
    if (nnue::is_ready()) {
        ensure_accumulator(pos, ss, ply);
        return nnue::evaluate_from_pair(ss[ply].acc, pos.side_to_move());
    }

    return evaluate(pos);
}

static inline void seed_root_accumulator(const Position& pos, SearchStack* ss)
{
    ss[0].acc_valid = false;
    if (!nnue::is_ready())
        return;

    nnue::refresh_acc(pos, WHITE, ss[0].acc.white);
    nnue::refresh_acc(pos, BLACK, ss[0].acc.black);
    ss[0].acc_valid = true;
}

static int qsearch(Position& pos, int alpha, int beta, int ply, SearchStack* ss)
{
    if (ply > seldepth) seldepth = ply;
    nodes_count++;

    const uint64_t pollMaskQ = tm.poll_interval_mask();
    if ((nodes_count & pollMaskQ) == 0)
        update_time();

    if (stop_flag)
        return alpha;

    if (pos.halfmove_clock() >= 100 || has_insufficient_material(pos) || (ply > 0 && pos.is_repetition_draw(ply)))
        return draw_score();

    if (ply >= MAX_PLY - 1)
        return eval_from_stack(pos, ss, ply);

    bool inChk = in_check(pos, pos.side_to_move());
    if (inChk && ply > QS_MAX_CHECK_PLY)
        return eval_from_stack(pos, ss, ply);

    Move q_tt_move = 0;
    TTEntry* qtt = tt_probe(pos.hash());
    int q_tt_score = 0;
    bool has_q_tt_score = false;
    if (qtt) {
        q_tt_move = qtt->best_move;
        q_tt_score = qtt->score;
        if (q_tt_score > MATE_SCORE - 1000) q_tt_score -= ply;
        else if (q_tt_score < -MATE_SCORE + 1000) q_tt_score += ply;
        has_q_tt_score = true;

        if (qtt->flag == TT_EXACT)
            return q_tt_score;
        if (qtt->flag == TT_ALPHA && q_tt_score <= alpha)
            return q_tt_score;
        if (qtt->flag == TT_BETA && q_tt_score >= beta)
            return q_tt_score;
    }

    int stand_pat = -INF;
    if (!inChk)
    {
        int raw_eval = qtt ? qtt->static_eval : eval_from_stack(pos, ss, ply);
        stand_pat = raw_eval + get_eval_correction(pos, ss, ply);

        int rule50 = pos.halfmove_clock();
        if (rule50 > 0) {
            stand_pat = stand_pat * (100 - rule50) / 100;
        }

        stand_pat = std::clamp(stand_pat, -MATE_SCORE + 1000, MATE_SCORE - 1000);

        ss[ply].raw_eval = raw_eval;
        ss[ply].static_eval = stand_pat;

        if (has_q_tt_score) {
            if ((qtt->flag == TT_BETA && q_tt_score > stand_pat)
                || (qtt->flag == TT_ALPHA && q_tt_score < stand_pat)
                || qtt->flag == TT_EXACT)
                stand_pat = q_tt_score;
        }

        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    MovePicker picker;
    picker.init_qsearch(pos, inChk, q_tt_move);

    int legal_moves = 0;

    for (Move m = picker.next(false); m; m = picker.next(false))
    {
        Piece movedPiece = pos.piece_on(from_sq(m));

        bool isQuiet = !is_capture(m) && !is_promotion(m);

        if (!inChk && isQuiet) {
            continue;
        }

        if (!inChk && is_capture(m) && !is_promotion(m)) {
            if (!movepick_see_ge(pos, m, -73)) {
                continue;
            }
        }

        if (!pos.make_move(m))
            continue;

        ss[ply + 1].acc_valid = false;

        tt_prefetch(pos.hash());
        ss[ply].current_move = m;
        ss[ply].moved_piece = movedPiece;
        legal_moves++;

        int score = -qsearch(pos, -beta, -alpha, ply + 1, ss);
        pos.undo_move();

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    if (inChk && legal_moves == 0)
        return -MATE_SCORE + ply;

    return alpha;
}


template <NodeType NT>
static int negamax(Position& pos, int depth, int alpha, int beta, int ply, SearchStack* ss)
{
    constexpr bool isPV = (NT == Root || NT == PV);
    constexpr bool isRoot = (NT == Root);

    if (ply > seldepth) seldepth = ply;

    if (stop_flag) return alpha;
    if (ply >= MAX_PLY - 1) return eval_from_stack(pos, ss, ply);
    nodes_count++;

    const uint64_t pollMaskN = tm.poll_interval_mask();
    if ((nodes_count & pollMaskN) == 0)
        update_time();

    if (pos.halfmove_clock() >= 100 || has_insufficient_material(pos) || (ply > 0 && pos.is_repetition_draw(ply)))
    {
        return draw_score();
    }

    alpha = std::max(alpha, -MATE_SCORE + ply);
    beta = std::min(beta, MATE_SCORE - ply - 1);
    if (alpha >= beta) return alpha;

    pv_length[ply] = 0;

    uint64_t key = pos.hash();
    int      original_alpha = alpha;
    TTEntry* entry = tt_probe(key);
    Move     tt_move = (entry) ? entry->best_move : 0;
    ss[ply].tt_pv = entry && entry->flag == TT_EXACT;
    ss[ply].seen_moves = 0;
    ss[ply].played_capture = false;
    ss[ply].cutoff_count = 0;
    ss[ply].hist_score = 0;

    int tt_score = 0;
    if (entry) {
        tt_score = entry->score;
        if (tt_score > MATE_SCORE - 1000) tt_score -= ply;
        else if (tt_score < -MATE_SCORE + 1000) tt_score += ply;
    }

    if (entry && entry->depth >= depth && ss[ply].excluded_move == 0)
    {
        if (entry->flag == TT_EXACT && !isPV && ply > 0)
            return tt_score;
        if (entry->flag == TT_ALPHA && tt_score <= alpha) return tt_score;
        if (entry->flag == TT_BETA && tt_score >= beta)  return tt_score;
    }

    bool inChk = in_check(pos, pos.side_to_move());

    if (inChk && depth <= 0)
        depth = 1;

    if (depth <= 0 && !inChk)
        return qsearch(pos, alpha, beta, ply, ss);

    // IIR
    if (depth >= 4 && tt_move == 0 && ss[ply].excluded_move == 0) {
        depth--;
    }

    int raw_eval = entry ? entry->static_eval : eval_from_stack(pos, ss, ply);
    int staticEval = raw_eval;

    if (!inChk) {
        staticEval += get_eval_correction(pos, ss, ply);

        int rule50 = pos.halfmove_clock();
        if (rule50 > 0) {
            staticEval = staticEval * (100 - rule50) / 100;
        }

        staticEval = std::clamp(staticEval, -MATE_SCORE + 1000, MATE_SCORE - 1000);
    }

    ss[ply].raw_eval = raw_eval;
    ss[ply].static_eval = staticEval;

    // --- Improving Status ---
    bool improving = false;
    if (!inChk) {
        if (ply >= 2 && ss[ply - 2].static_eval != -INF)
            improving = staticEval > ss[ply - 2].static_eval;
        else if (ply >= 4 && ss[ply - 4].static_eval != -INF)
            improving = staticEval > ss[ply - 4].static_eval;
    }


    // SINGULAR EXTENSIONS
    int extension = 0;
    if (ply > 0
        && depth >= 8
        && tt_move != 0
        && ss[ply].excluded_move == 0
        && entry
        && entry->depth >= depth - 3
        && entry->flag != TT_ALPHA
        && std::abs(tt_score) < MATE_SCORE - MAX_PLY)
    {
        int singular_margin = depth * 2;
        int singular_beta = tt_score - singular_margin;

        ss[ply].excluded_move = tt_move;
        int singular_score = negamax<NonPV>(pos, depth / 2, singular_beta - 1, singular_beta, ply, ss);
        ss[ply].excluded_move = 0;

        if (singular_score < singular_beta) {
            extension = 1;
        }
    }

    if constexpr (!isPV) {
        // --- RFP ---
        if (!inChk && std::abs(beta) < MATE_SCORE - MAX_PLY && depth <= 3 && ss[ply].excluded_move == 0)
        {
            int rfp_margin = 120 * depth;
            if (staticEval - rfp_margin >= beta && staticEval < 10000)
                return staticEval;
        }

        // --- NMP ---
        bool prev_is_null = (ply > 0 && ss[ply - 1].current_move == 0);
        if (!inChk && !prev_is_null && ss[ply].excluded_move == 0 && depth >= 4 && staticEval >= beta && staticEval < 10000 && has_non_pawn_material(pos, pos.side_to_move()))
        {
            int R = 3 + depth / 6;

            ss[ply].current_move = 0;
            ss[ply].moved_piece = NO_PIECE;

            pos.do_null_move();
            ss[ply + 1].acc_valid = false;
            int null_score = -negamax<NonPV>(pos, depth - R - 1, -beta, -beta + 1, ply + 1, ss);
            pos.undo_null_move();

            if (null_score >= beta)
            {
                return null_score >= MATE_SCORE - MAX_PLY ? beta : null_score;
            }
        }

        // --- ProbCut ---
        if (!inChk && depth >= 5 && std::abs(beta) < MATE_SCORE - MAX_PLY && ss[ply].excluded_move == 0)
        {
            int probcut_beta = beta + 200 - (improving ? 50 : 0);
            int pc_count = 0;

            MovePicker pc_picker;
            pc_picker.init_qsearch(pos, inChk, tt_move);

            for (Move m = pc_picker.next(false); m; m = pc_picker.next(false))
            {
                if (pc_count++ >= 3) break;

                if (movepick_see_ge(pos, m, 0))
                {
                    Piece movedPiece = pos.piece_on(from_sq(m));
                    if (pos.make_move(m))
                    {
                        ss[ply + 1].acc_valid = false;
                        ss[ply].current_move = m;
                        ss[ply].moved_piece = movedPiece;

                        int score = -qsearch(pos, -probcut_beta, -probcut_beta + 1, ply + 1, ss);

                        if (score >= probcut_beta)
                        {
                            score = -negamax<NonPV>(pos, depth - 4, -probcut_beta, -probcut_beta + 1, ply + 1, ss);
                        }

                        pos.undo_move();

                        if (score >= probcut_beta)
                        {
                            int store_score = score >= MATE_SCORE - MAX_PLY ? beta : score;
                            tt_store(key, depth - 3, store_score, TT_BETA, m, raw_eval);

                            ss[ply].current_move = 0;
                            ss[ply].moved_piece = NO_PIECE;
                            return store_score;
                        }
                    }
                }
            }
            ss[ply].current_move = 0;
            ss[ply].moved_piece = NO_PIECE;
        }
    }

    Move counter_move = 0;
    if (ply >= 1)
    {
        const Move prev = ss[ply - 1].current_move;
        if (prev)
            counter_move = countermove[from_sq(prev)][to_sq(prev)];
    }

    const MovePicker::MainOrderData orderData = build_main_order_data(ss, ply);
    MovePicker picker;
    picker.init_main(pos,
        tt_move,
        killer_moves[0][ply],
        killer_moves[1][ply],
        counter_move,
        &orderData);

    int  legal_moves = 0;
    int  moveCount = 0;
    Move best_move = 0;

    Move quiet_searched[256];
    Piece quiet_pieces[256];
    int  quiet_count = 0;

    Move capture_searched[256];
    Piece capture_attackers[256];
    Piece capture_victims[256];
    int  capture_count = 0;

    for (Move m = picker.next(false); m; m = picker.next(false))
    {
        if (m == ss[ply].excluded_move)
            continue;

        bool isQuiet = !is_capture(m) && !is_promotion(m);

        Piece movedPiece = pos.piece_on(from_sq(m));
        Piece capturedPiece = captured_piece_for_move(pos, m);

        int hist_score = 0;
        if (isQuiet) {
            hist_score = quiet_history_score(pos, m, ply, ss);
        }
        else {
            hist_score = capture_history_score(pos, m);
        }
        ss[ply].hist_score = hist_score;

        if (!inChk && isQuiet && depth <= 8 && std::abs(alpha) < MATE_SCORE - MAX_PLY) {
            int lmp_threshold = 3 + depth * depth / 2;
            if (moveCount >= lmp_threshold) {
                continue;
            }
        }

        if constexpr (!isPV) {
            if (!inChk && isQuiet && depth <= 8 && std::abs(alpha) < MATE_SCORE - MAX_PLY) {
                int fp_margin = 120 * depth;
                if (m != killer_moves[0][ply] && m != killer_moves[1][ply] && hist_score < 4000) {
                    if (staticEval + fp_margin <= alpha) {
                        continue;
                    }
                }
            }
        }

        if (depth <= 8 && !inChk) {
            int see_threshold = isQuiet ? -50 * depth : -100 * depth;
            if (!movepick_see_ge(pos, m, see_threshold)) {
                continue;
            }
        }

        if (!pos.make_move(m))
            continue;

        ss[ply + 1].acc_valid = false;

        bool givesChk = pos.gives_check(m);

        tt_prefetch(pos.hash());

        ss[ply].current_move = m;
        ss[ply].moved_piece = movedPiece;

        legal_moves++;
        moveCount++;
        ss[ply].seen_moves = moveCount;
        ss[ply].played_capture = !isQuiet;

        if (isQuiet && quiet_count < 256)
        {
            quiet_searched[quiet_count] = m;
            quiet_pieces[quiet_count] = movedPiece;
            quiet_count++;
        }
        else if (!isQuiet && capture_count < 256)
        {
            capture_searched[capture_count] = m;
            capture_attackers[capture_count] = movedPiece;
            capture_victims[capture_count] = capturedPiece;
            capture_count++;
        }

        int  score;
        int ext = (m == tt_move) ? extension : 0;

        if (givesChk && moveCount == 1) {
            ext = 1;
        }

        int searchedDepth = std::min(MAX_PLY - 1, depth - 1 + ext);

        if (moveCount == 1) {
            pv_length[ply + 1] = 0;
            score = -negamax<isPV ? PV : NonPV>(pos, searchedDepth, -beta, -alpha, ply + 1, ss);
        }
        else {
            pv_length[ply + 1] = 0;

            // LMR
            int R = 0;

            if (depth >= 3 && moveCount > (isPV ? 3 : 1) && !inChk && ext == 0)
            {
                int d = std::min(depth, 63);
                int c = std::min(moveCount, 255);

                R = LMR_TABLE[isQuiet ? 1 : 0][d][c];

                if constexpr (isPV) R -= 1;
                else R += 1;


                if (isQuiet) {

                    R -= std::clamp(hist_score / 8192, -3, 3);


                    if (m == killer_moves[0][ply] || m == killer_moves[1][ply]) {
                        R -= 1;
                    }

                    if (ply >= 1 && ss[ply - 1].current_move != 0) {
                        Move prev = ss[ply - 1].current_move;
                        if (m == countermove[from_sq(prev)][to_sq(prev)]) {
                            R -= 1;
                        }
                    }
                }
                else {
                    R -= std::clamp(hist_score / 8192, -2, 2);
                }

                if (!improving) {
                    R += 1;
                }

                if (tt_move == 0) {
                    R += 1;
                }

                R = std::clamp(R, 0, searchedDepth - 1);
            }

            if (R > 0) {
                score = -negamax<NonPV>(pos, searchedDepth - R, -(alpha + 1), -alpha, ply + 1, ss);


                if (score > alpha) {
                    int new_depth = searchedDepth;

                    if (score > alpha + 50) {
                        new_depth += 1;
                    }

                    score = -negamax<NonPV>(pos, new_depth, -(alpha + 1), -alpha, ply + 1, ss);
                }
            }
            else {
                score = -negamax<NonPV>(pos, searchedDepth, -(alpha + 1), -alpha, ply + 1, ss);
            }

            if constexpr (isPV) {
                if (score > alpha && score < beta) {
                    pv_length[ply + 1] = 0;
                    score = -negamax<PV>(pos, searchedDepth, -beta, -alpha, ply + 1, ss);
                }
            }
        }

        pos.undo_move();

        if (stop_flag) break;

        if (score >= beta)
        {
            ss[ply].cutoff_count += 1;
            if (isQuiet)
            {
                killer_moves[1][ply] = killer_moves[0][ply];
                killer_moves[0][ply] = m;

                int& h = history_heuristic[from_sq(m)][to_sq(m)];
                h += depth * depth;
                if (h > HISTORY_CLAMP) h = HISTORY_CLAMP;

                if (ply >= 1)
                {
                    Move prev = ss[ply - 1].current_move;
                    if (prev)
                        countermove[from_sq(prev)][to_sq(prev)] = m;
                }

                update_continuation_histories(ss, ply, m, movedPiece, depth * depth);
            }
            else
            {
                update_capture_history(movedPiece, capturedPiece, depth * depth);
            }

            int store_score = beta;
            if (store_score > MATE_SCORE - 1000) store_score += ply;
            else if (store_score < -MATE_SCORE + 1000) store_score -= ply;

            if (ss[ply].excluded_move == 0) {
                tt_store(key, depth, store_score, TT_BETA, m, raw_eval);
            }

            if (!inChk && !is_capture(m) && std::abs(beta) < MATE_SCORE - 1000 && beta > staticEval) {
                if (ss[ply].excluded_move == 0) {
                    update_eval_correction(pos, ss, ply, depth, beta - staticEval);
                }
            }

            return beta;
        }

        if (score > alpha)
        {
            alpha = score;
            best_move = m;

            pv_table[ply][0] = m;
            int child_len = pv_length[ply + 1];
            if (child_len < 0 || child_len > MAX_PLY - ply - 1) child_len = 0;
            for (int j = 0; j < child_len; ++j)
                pv_table[ply][j + 1] = pv_table[ply + 1][j];
            pv_length[ply] = child_len + 1;
        }
    }
    if (stop_flag)
        return alpha;

    if (legal_moves == 0)
        return inChk ? -MATE_SCORE + ply : 0;

    if (!stop_flag)
    {
        const int bonus = depth * depth;

        auto clamp_hist = [](int& x)
            {
                if (x > HISTORY_CLAMP) x = HISTORY_CLAMP;
                else if (x < -HISTORY_CLAMP) x = -HISTORY_CLAMP;
            };

        const bool best_is_quiet = best_move && !is_capture(best_move) && !is_promotion(best_move);
        const bool best_is_capture = best_move && is_capture(best_move);

        for (int i = 0; i < quiet_count; ++i)
        {
            Move qm = quiet_searched[i];
            Piece qp = quiet_pieces[i];
            if (!qm || qp == NO_PIECE) continue;

            const bool isBest = (best_is_quiet && qm == best_move);
            const int delta = isBest ? bonus : -bonus;

            int& h = history_heuristic[from_sq(qm)][to_sq(qm)];
            h += delta;
            clamp_hist(h);

            update_continuation_histories(ss, ply, qm, qp, delta);

            if (isBest && ply >= 1)
            {
                Move prev = ss[ply - 1].current_move;
                if (prev)
                    countermove[from_sq(prev)][to_sq(prev)] = qm;
            }
        }

        for (int i = 0; i < capture_count; ++i)
        {
            Move cm = capture_searched[i];
            Piece attacker = capture_attackers[i];
            Piece victim = capture_victims[i];
            if (!cm || attacker == NO_PIECE || victim == NO_PIECE) continue;

            const bool isBest = (best_is_capture && cm == best_move);
            const int delta = isBest ? bonus : -bonus;
            update_capture_history(attacker, victim, delta);
        }
    }

    TTFlag flag;
    if (alpha <= original_alpha) flag = TT_ALPHA;
    else if (alpha >= beta)      flag = TT_BETA;
    else                         flag = TT_EXACT;

    int store_score = alpha;
    if (store_score > MATE_SCORE - 1000) store_score += ply;
    else if (store_score < -MATE_SCORE + 1000) store_score -= ply;

    if (ss[ply].excluded_move == 0) {
        tt_store(key, depth, store_score, flag, best_move, raw_eval);
    }

    if (!inChk && !(best_move && is_capture(best_move)) && std::abs(alpha) < MATE_SCORE - 1000) {
        bool skip_update = false;
        if (ss[ply].excluded_move != 0) skip_update = true;
        if (flag == TT_ALPHA && alpha >= staticEval) skip_update = true;
        if (flag == TT_BETA && alpha <= staticEval) skip_update = true;

        if (!skip_update) {
            update_eval_correction(pos, ss, ply, depth, alpha - staticEval);
        }
    }

    return alpha;
}

static inline double clampd(double x, double lo, double hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline double compute_time_scale(int bestScore, int secondScore, int prevScore,
    bool pvChanged, uint64_t bestMoveNodes, uint64_t totalNodes,
    int currentDepth, int depth1Score, bool hasDepth1Score)
{
    double u = 0.0;

    int swing = std::abs(bestScore - prevScore);
    u += std::min(1.0, swing / 80.0);

    if (pvChanged) u += 0.35;

    if (secondScore > -INF / 2)
    {
        int gap = bestScore - secondScore;
        if (gap < 40) u += (40 - gap) / 40.0;
    }

    if (totalNodes > 0)
    {
        double frac = double(bestMoveNodes) / double(totalNodes);
        if (frac < 0.30)      u += 0.35;
        else if (frac < 0.45) u += 0.15;
        else if (frac > 0.65) u -= 0.10;
    }

    if (hasDepth1Score && currentDepth > 1)
    {
        double drift = std::abs(bestScore - depth1Score);
        u += clampd(drift / 160.0, 0.0, 0.35);
    }

    double scale = 0.60 + 1.60 * clampd(u / 2.0, 0.0, 1.0);

    return clampd(scale, 0.50, 2.50);
}

struct RootMoveState {
    Move move = 0;
    int prev_score = 0;
    int avg_score = 0;
    bool has_score = false;
    int sort_score = 0;
};

static void seed_root_sort_scores(Position& pos,
    RootMoveState* root_states,
    int root_count,
    Move last_best_move,
    int depth)
{
    for (int i = 0; i < root_count; ++i)
    {
        int score = score_root_move(pos, root_states[i].move, last_best_move);
        if (root_states[i].has_score)
        {
            if (depth <= 8) {
                score += std::clamp(root_states[i].prev_score, -160, 160) * 48;
                score += std::clamp(root_states[i].avg_score, -160, 160) * 24;
            }
            else if (depth <= 12) {
                score += std::clamp(root_states[i].prev_score, -160, 160) * 24;
                score += std::clamp(root_states[i].avg_score, -160, 160) * 12;
            }
        }
        root_states[i].sort_score = score;
    }
}

static void update_root_state_scores(RootMoveState* root_states,
    int root_count,
    const int* final_root_scores,
    const bool* final_root_scored)
{
    for (int i = 0; i < root_count; ++i)
    {
        if (!final_root_scored[i])
            continue;

        const int score = final_root_scores[i];
        if (root_states[i].has_score)
            root_states[i].avg_score = (root_states[i].avg_score * 3 + score) / 4;
        else
            root_states[i].avg_score = score;

        root_states[i].prev_score = score;
        root_states[i].has_score = true;
    }
}

static void initialize_search_stack(SearchStack* ss)
{
    for (int i = 0; i < MAX_PLY + 4; ++i)
    {
        ss[i].current_move = 0;
        ss[i].moved_piece = NO_PIECE;
        ss[i].static_eval = -INF;
        ss[i].raw_eval = -INF;
        ss[i].seen_moves = 0;
        ss[i].played_capture = false;
        ss[i].cutoff_count = 0;
        ss[i].tt_pv = false;
        ss[i].hist_score = 0;
        ss[i].excluded_move = 0;
        ss[i].acc_valid = false;
    }
}

static void generate_legal_root_moves(Position& pos, MoveList& root_moves)
{
    MoveList pseudo_legal_root_moves;
    generate_moves(pos, pseudo_legal_root_moves);

    root_moves.size = 0;
    for (int i = 0; i < pseudo_legal_root_moves.size; ++i)
    {
        Move m = pseudo_legal_root_moves.moves[i];
        if (pos.make_move(m))
        {
            root_moves.push(m);
            pos.undo_move();
        }
    }
}

static void print_search_info(Position& pos, int depth, int best_score)
{
    if (g_silent_search)
        return;

    int time_spent = tm.elapsed();
    uint64_t nps = (time_spent > 0)
        ? (nodes_count * 1000ULL / time_spent)
        : 0;

    std::cout << "info depth " << depth << " seldepth " << seldepth;

    if (best_score > MATE_SCORE - 1000)
    {
        int moves_to_mate = (MATE_SCORE - best_score + 1) / 2;
        std::cout << " score mate " << moves_to_mate;
    }
    else if (best_score < -MATE_SCORE + 1000)
    {
        int moves_to_mate = -(MATE_SCORE + best_score) / 2;
        std::cout << " score mate " << moves_to_mate;
    }
    else
    {
        std::cout << " score cp " << best_score;
    }

    std::cout << " nodes " << nodes_count
        << " nps " << nps
        << " hashfull " << tt_hashfull()
        << " time " << time_spent
        << " pv ";

    int pv_made = 0;
    for (int i = 0; i < pv_length[0]; ++i)
    {
        Move mv = pv_table[0][i];
        if (!pos.make_move(mv))
            break;
        ++pv_made;

        Square f = from_sq(mv);
        Square t = to_sq(mv);
        std::cout << char('a' + file_of(f)) << char('1' + rank_of(f))
            << char('a' + file_of(t)) << char('1' + rank_of(t));
        if (is_promotion(mv))
        {
            PieceType pt = promotion_type(mv);
            if (pt == QUEEN)  std::cout << "q";
            else if (pt == ROOK)   std::cout << "r";
            else if (pt == BISHOP) std::cout << "b";
            else if (pt == KNIGHT) std::cout << "n";
        }
        std::cout << " ";
    }
    while (pv_made-- > 0)
        pos.undo_move();
    std::cout << std::endl;
}

SearchResult search(Position& pos, int max_depth, int movetime, int time_left, int increment, int moves_to_go, bool infinite, uint64_t max_nodes)
{
    stop_flag = false;
    nodes_count = 0;
    target_max_nodes = max_nodes;
    seldepth = 0;

    std::unique_ptr<SearchStack[]> ss_storage = std::make_unique<SearchStack[]>(MAX_PLY + 4);
    SearchStack* ss = ss_storage.get();
    initialize_search_stack(ss);
    seed_root_accumulator(pos, ss);

    for (int i = 0; i < MAX_PLY; ++i) pv_length[i] = 0;
    for (int i = 0; i < MAX_PLY; ++i)
    {
        killer_moves[0][i] = 0;
        killer_moves[1][i] = 0;
    }

    decay_history_tables_if_needed();

    tm.reset();
    tt_new_search();

    int current_fullmove = 30;

    tm.set_limits(
        infinite,
        movetime,
        time_left,
        increment,
        moves_to_go,
        current_fullmove
    );

    tm.start();

    SearchResult result{};
    result.best_move = 0;
    result.score = 0;
    result.depth = 0;
    result.nodes = 0;

    MoveList root_moves;
    generate_legal_root_moves(pos, root_moves);
    RootMoveState root_states[256];
    const int root_count = root_moves.size;
    for (int i = 0; i < root_count; ++i)
        root_states[i].move = root_moves.moves[i];

    if (root_count == 0)
    {
        result.best_move = 0;
        result.score = in_check(pos, pos.side_to_move()) ? -MATE_SCORE : 0;
        result.depth = 0;
        result.nodes = nodes_count;
        return result;
    }

    if (pos.halfmove_clock() >= 100 || has_insufficient_material(pos) || pos.is_repetition_draw(0))
    {
        result.best_move = root_states[0].move;
        result.score = draw_score();
        result.depth = 0;
        result.nodes = nodes_count;
        return result;
    }
    result.best_move = root_states[0].move;
    Move last_best_move = 0;
    int  prev_score = 0;
    int  depth1_score = 0;
    bool has_depth1_score = false;

    for (int depth = 1; depth <= max_depth; ++depth)
    {
        update_time();
        if (stop_flag)
            break;

        const uint64_t depth_nodes_start = nodes_count;

        seed_root_sort_scores(pos, root_states, root_count, last_best_move, depth);

        std::stable_sort(root_states, root_states + root_count,
            [](const RootMoveState& a, const RootMoveState& b) {
                return a.sort_score > b.sort_score;
            });

        int delta = 25;
        int alpha = -INF;
        int beta = INF;

        if (depth >= 5) {
            alpha = std::max(-INF, prev_score - delta);
            beta = std::min(INF, prev_score + delta);
        }

        int best_score = -INF;
        int best2_final = -INF;
        Move current_depth_best_move = 0;
        uint64_t current_depth_best_move_nodes = 0;

        int final_root_scores[256]{};
        bool final_root_scored[256]{};

        while (true) {
            int search_alpha = alpha;
            int search_best = -INF;
            int search_second = -INF;
            Move search_best_move = 0;
            uint64_t search_best_move_nodes = 0;
            pv_length[0] = 0;

            std::fill(final_root_scored, final_root_scored + root_count, false);

            for (int i = 0; i < root_count; ++i)
            {
                if ((i & 3) == 0)
                {
                    update_time();
                    if (stop_flag)
                        break;
                }

                Move m = root_states[i].move;
                Piece movedPiece = pos.piece_on(from_sq(m));
                if (!pos.make_move(m))
                    continue;

                ss[1].acc_valid = false;

                tt_prefetch(pos.hash());

                ss[0].current_move = m;
                ss[0].moved_piece = movedPiece;

                pv_length[1] = 0;

                const uint64_t move_nodes_before = nodes_count;
                int score;

                if (i == 0) {
                    score = -negamax<PV>(pos, depth - 1, -beta, -search_alpha, 1, ss);
                }
                else {
                    score = -negamax<NonPV>(pos, depth - 1, -search_alpha - 1, -search_alpha, 1, ss);

                    if (score > search_alpha && score < beta) {
                        score = -negamax<PV>(pos, depth - 1, -beta, -search_alpha, 1, ss);
                    }
                }

                pos.undo_move();
                const uint64_t move_nodes = nodes_count - move_nodes_before;
                final_root_scores[i] = score;
                final_root_scored[i] = true;

                if (stop_flag) break;

                if (score > search_best)
                {
                    search_second = search_best;
                    search_best = score;
                    search_best_move = m;
                    search_best_move_nodes = move_nodes;
                }
                else if (score > search_second)
                {
                    search_second = score;
                }
                if (score > search_alpha)
                {
                    search_alpha = score;

                    pv_table[0][0] = m;
                    int child_len = pv_length[1];
                    if (child_len > 0 && child_len < MAX_PLY)
                    {
                        for (int j = 0; j < child_len; ++j)
                            pv_table[0][j + 1] = pv_table[1][j];
                        pv_length[0] = child_len + 1;
                    }
                    else
                    {
                        pv_length[0] = 1;
                    }
                }
            }

            if (stop_flag) break;

            best_score = search_best;

            if (best_score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(-INF, alpha - delta);
                delta += delta / 2;
            }
            else if (best_score >= beta) {
                beta = std::min(INF, beta + delta);
                delta += delta / 2;
            }
            else {
                best2_final = search_second;
                current_depth_best_move = search_best_move;
                current_depth_best_move_nodes = search_best_move_nodes;
                break;
            }
        }

        if (stop_flag) break;

        if (current_depth_best_move != 0)
        {
            update_root_state_scores(root_states, root_count, final_root_scores, final_root_scored);

            bool pvChanged = (current_depth_best_move != last_best_move);
            last_best_move = current_depth_best_move;
            result.best_move = last_best_move;
            result.score = best_score;
            result.depth = depth;
            result.nodes = nodes_count;
            int prev_score_old = prev_score;
            prev_score = best_score;
            if (depth == 1)
            {
                depth1_score = best_score;
                has_depth1_score = true;
            }

            if (!infinite && movetime <= 0 && time_left > 0)
            {
                const uint64_t depth_nodes_total = std::max<uint64_t>(1, nodes_count - depth_nodes_start);
                double scale = compute_time_scale(best_score,
                    best2_final,
                    prev_score_old,
                    pvChanged,
                    current_depth_best_move_nodes,
                    depth_nodes_total,
                    depth,
                    depth1_score,
                    has_depth1_score);
                tm.set_dynamic_scale(scale);
            }
        }

        print_search_info(pos, depth, best_score);

        if (best_score >= MATE_SCORE - 4 || best_score <= -MATE_SCORE + 4)
            break;

        if (target_max_nodes > 0 && nodes_count >= target_max_nodes) {
            break;
        }
    }

    return result;
}