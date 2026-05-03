#include "datagen.h"
#include "position.h"
#include "movegen.h"
#include "search.h"
#include "attacks.h"
#include "tt.h"

#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <chrono> 

namespace Throne {
    namespace Datagen {


#pragma pack(push, 1)
        struct BulletFormatData {
            uint64_t occupancy;      
            uint8_t  pieces[16];     
            int16_t  score;          
            uint8_t  wdl;           
            uint8_t  our_ksq;        
            uint8_t  opp_ksq;        
            uint8_t  stm_temp;       
            uint8_t  padding[2];     
        };
#pragma pack(pop)


        static MoveList get_legal_moves(Position& pos) {
            MoveList pseudo, legal;
            generate_moves(pos, pseudo);
            for (int i = 0; i < pseudo.size; ++i) {
                Move m = pseudo.moves[i];
                if (pos.make_move(m)) {
                    legal.push(m);
                    pos.undo_move();
                }
            }
            return legal;
        }


        void run(const std::string& output_path, const std::string& epd_path, int nodes_per_move, int target_games) {

            g_silent_search = true;
            std::mt19937 rng(std::random_device{}());

            std::vector<std::string> opening_fens;
            bool use_epd = false;

            if (epd_path != "none") {
                std::ifstream epd_file(epd_path);
                if (epd_file) {
                    std::string line;
                    while (std::getline(epd_file, line)) {
                        if (!line.empty()) opening_fens.push_back(line);
                    }
                    epd_file.close();
                    if (!opening_fens.empty()) {
                        use_epd = true;
                        std::cerr << "Loaded " << opening_fens.size() << " openings from " << epd_path << "\n";
                    }
                }
                else {
                    std::cerr << "Warning: Could not open EPD file (" << epd_path << "). Falling back to random moves.\n";
                }
            }
            else {
                std::cerr << "EPD set to 'none'. Using standard random opening.\n";
            }


            std::ofstream out(output_path, std::ios::binary | std::ios::app);
            if (!out) {
                std::cerr << "Datagen: failed to open " << output_path << "\n";
                return;
            }

            tt_resize_mb(16); 

            int games_played = 0;
            int positions_saved = 0;

            std::cerr << "Starting Datagen: " << target_games << " games at " << nodes_per_move << " Nodes/Move\n";


            auto total_start_time = std::chrono::steady_clock::now();
            auto batch_start_time = std::chrono::steady_clock::now();

            std::uniform_int_distribution<size_t> epd_dist(0, opening_fens.empty() ? 0 : opening_fens.size() - 1);

            while (games_played < target_games) {
                Position pos;
                bool valid = true;

                if (use_epd) {
                    pos.set_fen(opening_fens[epd_dist(rng)]);
                }
                else {
                    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                    std::uniform_int_distribution<int> openDist(8, 9);
                    int openPlies = openDist(rng);

                    for (int i = 0; i < openPlies; ++i) {
                        MoveList moves = get_legal_moves(pos);
                        if (moves.size == 0) { valid = false; break; }
                        std::uniform_int_distribution<int> md(0, moves.size - 1);
                        pos.make_move(moves.moves[md(rng)]);
                    }
                }

                if (!valid) continue;

                std::vector<BulletFormatData> positions;
                positions.reserve(100);

                int result = 1; // 1 = Draw, 2 = White Wins, 0 = Black Wins

                clear_search_state_for_new_game();
                tt_clear();


                for (int ply = 0; ply < 400; ++ply) {
                    MoveList moves = get_legal_moves(pos);

                    
                    if (moves.size == 0) {
                        if (in_check(pos, pos.side_to_move())) {
                            result = (pos.side_to_move() == WHITE) ? 0 : 2;
                        }
                        else {
                            result = 1;
                        }
                        break;
                    }


                    if (pos.halfmove_clock() >= 100) { result = 1; break; }
                    if (pos.is_repetition_draw(ply)) { result = 1; break; }

                    SearchResult res = search(pos, 64, 9999999, 9999999, 0, 0, false, nodes_per_move);

                    if (res.best_move == 0) break;

                    int score = res.score;


                    if (!in_check(pos, pos.side_to_move()) && std::abs(score) < 20000) {

                        BulletFormatData data = { 0 }; 

                        uint64_t occ = 0;
                        int piece_idx = 0;
                        uint8_t our_ksq = 0;
                        uint8_t opp_ksq = 0;

                        bool is_black_stm = (pos.side_to_move() == BLACK);

                        for (int i = 0; i < 64; ++i) {

                            int sq_idx = is_black_stm ? (i ^ 56) : i;
                            Square sq = Square(sq_idx);
                            Piece p = pos.piece_on(sq);

                            if (p != NO_PIECE) {
                                occ |= (1ULL << i);

                                uint8_t p_type;
                                switch (piece_type(p)) {
                                case PAWN:   p_type = 0; break;
                                case KNIGHT: p_type = 1; break;
                                case BISHOP: p_type = 2; break;
                                case ROOK:   p_type = 3; break;
                                case QUEEN:  p_type = 4; break;
                                case KING:   p_type = 5; break;
                                default:     p_type = 0; break;
                                }

                                uint8_t color_bit = (piece_color(p) == BLACK) ? 1 : 0;

                                if (is_black_stm) {
                                    color_bit ^= 1;
                                }

                                uint8_t piece_code = p_type | (color_bit << 3);

                                if (p_type == 5) {
                                    if (color_bit == 0) {
                                        our_ksq = i;       
                                    }
                                    else {
                                        opp_ksq = i ^ 56;  
                                    }
                                }

                                if (piece_idx % 2 == 0) {
                                    data.pieces[piece_idx / 2] |= piece_code;
                                }
                                else {
                                    data.pieces[piece_idx / 2] |= (piece_code << 4);
                                }
                                piece_idx++;
                            }
                        }

                        data.occupancy = occ;
                        data.our_ksq = our_ksq;
                        data.opp_ksq = opp_ksq;

                        data.score = score;

                        data.stm_temp = is_black_stm ? 1 : 0;

                        positions.push_back(data);
                    }


                    if (std::abs(score) >= 20000) {
                        result = (score > 0) ? ((pos.side_to_move() == WHITE) ? 2 : 0)
                            : ((pos.side_to_move() == WHITE) ? 0 : 2);
                        break;
                    }

                    pos.make_move(res.best_move);
                }


                for (auto& p : positions) {
                    if (result == 1) {
                        p.wdl = 1; 
                    }
                    else {
                        bool stm_is_black = (p.stm_temp == 1);
                        bool black_won = (result == 0); 


                        if (stm_is_black == black_won) {
                            p.wdl = 2; 
                        }
                        else {
                            p.wdl = 0; 
                        }
                    }


                    p.stm_temp = 0;

                    out.write(reinterpret_cast<const char*>(&p), sizeof(BulletFormatData));
                    positions_saved++;
                }

                games_played++;
                if (games_played % 10 == 0) {
                    auto batch_end_time = std::chrono::steady_clock::now();
                    std::chrono::duration<double> batch_elapsed = batch_end_time - batch_start_time;

                    std::cerr << "Played: " << games_played
                        << " games | Positions: " << positions_saved
                        << " | Time: " << batch_elapsed.count() << " s\n";

                    batch_start_time = std::chrono::steady_clock::now();
                }
            }

            auto total_end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;

            out.close();
            std::cerr << "Datagen complete! Saved " << positions_saved << " positions to " << output_path << "\n";
            std::cerr << "Total time elapsed: " << total_elapsed.count() << " seconds.\n";
        }

    } // namespace Datagen
} // namespace Throne