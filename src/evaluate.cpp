#include "evaluate.h"

#include "nnue.h"

int piece_value(PieceType pt)
{
    switch (pt)
    {
    case PAWN:   return 100;
    case KNIGHT: return 300;
    case BISHOP: return 300;
    case ROOK:   return 500;
    case QUEEN:  return 900;
    default:     return 0;
    }
}

int evaluate(const Position& pos)
{
    return nnue::is_ready() ? nnue::evaluate(pos) : 0;
}
