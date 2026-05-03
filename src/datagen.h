#pragma once
#include <string>

namespace Throne {
    namespace Datagen {
        void run(const std::string& output_path, const std::string& epd_path, int nodes_per_move, int target_games);
    }
}