#pragma once

#include "move.h"

class Position;


int movepick_capture_mvv_lva(const Position& pos, Move m);
bool movepick_see_ge(const Position& pos, Move m, int threshold);

class MovePicker {
public:
    struct MainOrderData {
        const int (*history)[64] = nullptr;
        const int (*capture_history)[16] = nullptr; 
        const int (*cont1)[64] = nullptr;
        const int (*cont2)[64] = nullptr;
        int quiet_check_bonus = 0;
    };

    MovePicker();

    static int score_main_move(const Position& pos, Move m, const MainOrderData& order_data);

    void init_main(const Position& pos,
                   Move tt_move,
                   Move killer1,
                   Move killer2,
                   Move counter_move,
                   const MainOrderData* order_data);


    void init_qsearch(const Position& pos,
                      bool in_check,
                      Move tt_move);

    Move next(bool skip_quiets = false);

private:
    enum Stage {
        ST_DONE,
        ST_TT,
        ST_GEN_CAPTURES,
        ST_CAPTURES,
        ST_KILLER1,
        ST_KILLER2,
        ST_COUNTER,
        ST_GEN_QUIETS,
        ST_QUIETS,
        ST_BAD_CAPTURES,
        ST_QS_TT,
        ST_QS_GEN,
        ST_QS_MOVES,
    };

    struct Bucket {
        Move moves[MAX_MOVES]{};
        int  scores[MAX_MOVES]{};
        int  count = 0;
        int  head = 0;

        void reset();
        void push(Move m, int s);
        bool has_next() const;
        Move pop_next(int& last_score_value, int fallback_score);
    };

    Stage stage = ST_DONE;

    const Position* pos_ptr = nullptr;
    MainOrderData order_data{};
    bool qs_in_check = false;

    Move tt = 0;
    Move killer_1 = 0;
    Move killer_2 = 0;
    Move counter = 0;

    bool has_tt = false;
    bool has_k1 = false;
    bool has_k2 = false;
    bool has_counter = false;

    bool captures_generated = false;
    bool quiets_generated = false;
    bool qs_generated = false;

    Bucket captures;
    Bucket bad_captures;
    Bucket quiets;
    Bucket qs_moves;

    int last_score_value = 0;
};


