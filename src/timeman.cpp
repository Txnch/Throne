#include "timeman.h"
#include <algorithm>
#include <cmath> 

namespace {


    static inline int clampi(int x, int lo, int hi) {
        return std::max(lo, std::min(x, hi));
    }

} // namespace

TimeManager::TimeManager() {
    reset();
}

void TimeManager::reset() {
    infinite_mode = false;
    fixed_movetime = false;

    time_left = 0;
    increment = 0;
    moves_to_go = 0;
    fullmove_number = 1; 

    optimum_time = 0;
    maximum_time = 0;

    base_optimum_time = 0;
    base_maximum_time = 0;
}


void TimeManager::set_limits(bool infinite,
    int move_time,
    int timeLeft,
    int inc,
    int movesToGo,
    int fullmove) {
    infinite_mode = infinite;
    fixed_movetime = false;
    fullmove_number = std::max(1, fullmove); 

    if (infinite_mode) {
        optimum_time = maximum_time = 0;
        base_optimum_time = base_maximum_time = 0;
        return;
    }

    time_left = std::max(0, timeLeft);
    increment = std::max(0, inc);
    moves_to_go = movesToGo;

    if (move_time > 0) {
        fixed_movetime = true;

        const int overhead = clampi(std::max(2, move_time / 40), 2, 40);
        const int hard = std::max(1, move_time - overhead);
        const int soft = std::max(1, hard - std::max(1, overhead / 2));

        optimum_time = std::min(soft, hard);
        maximum_time = std::max(optimum_time, hard);
        maximum_time = std::min(maximum_time, std::max(1, move_time));

        base_optimum_time = optimum_time;
        base_maximum_time = maximum_time;
        return;
    }

    compute_time();

    base_optimum_time = optimum_time;
    base_maximum_time = maximum_time;
}

void TimeManager::compute_time() {
    if (time_left <= 0) {
        optimum_time = maximum_time = 0;
        return;
    }


    const int dynamic_safety = clampi(time_left / 80, 10, 250);

    const int inc_safety = increment / 4;
    const int safety_margin = dynamic_safety + inc_safety;
    const int usable_time = std::max(1, time_left - safety_margin);

    int soft = 0;
    int hard = 0;


    if (moves_to_go > 0) {

        int mtg = clampi(moves_to_go, 2, 64);
        int base = usable_time / mtg;

        base += (increment * 7) / 10;
        base = std::max(base, 1);

        soft = base;

        hard = std::max(soft + 1, (soft * 5) / 3);


        if (moves_to_go > 0 && moves_to_go <= 8)
            hard = std::max(hard, soft * 2);

    }
    else {

        double soft_scale = 0.024 + 0.042 * (1.0 - std::exp(-0.045 * fullmove_number));
        double hard_scale = 0.742; 

        soft = int(soft_scale * usable_time + 0.75 * increment);
        hard = int(hard_scale * usable_time + 0.75 * increment);
    }


    if (time_left < 2000) {
        double panic_factor = std::max(0.1, (double)time_left / 2000.0);

        soft = std::max(1, int(soft * panic_factor));
        hard = std::max(soft + 10, int(hard * (panic_factor + 0.2)));
    }


    if (increment > 0) {
        int min_soft_from_inc = (increment * 6) / 10;
        soft = std::max(soft, min_soft_from_inc);
        hard = std::max(hard, increment); 
    }

    soft = std::min(soft, usable_time);
    hard = std::min(hard, usable_time);

    optimum_time = std::max(1, soft);
    maximum_time = std::max(optimum_time + 1, hard);
    maximum_time = std::min(maximum_time, usable_time);
}

void TimeManager::set_dynamic_scale(double scale) {
    if (infinite_mode || fixed_movetime)
        return;

    if (base_optimum_time <= 0)
        return;

    if (scale < 0.50) scale = 0.50;
    if (scale > 2.50) scale = 2.50;

    const int max_cap = std::max(1, base_maximum_time);

    int scaled_soft = int(base_optimum_time * scale);
    scaled_soft = clampi(scaled_soft, 1, max_cap);

    double hard_scale = 0.70 + 0.30 * scale;
    if (hard_scale < 0.70) hard_scale = 0.70;
    if (hard_scale > 1.20) hard_scale = 1.20;

    int scaled_hard = int(base_maximum_time * hard_scale);
    scaled_hard = clampi(scaled_hard, 1, max_cap);

    const int min_hard = std::min(max_cap, scaled_soft + 1);

    optimum_time = scaled_soft;
    maximum_time = std::max(min_hard, scaled_hard);
    if (maximum_time < optimum_time)
        maximum_time = optimum_time;
}

void TimeManager::start() {
    start_time = std::chrono::steady_clock::now();
}

int TimeManager::elapsed() const {
    auto now = std::chrono::steady_clock::now();
    return int(std::chrono::duration_cast<std::chrono::milliseconds>(
        now - start_time
    ).count());
}

bool TimeManager::should_stop(uint64_t /*nodes*/) {
    if (infinite_mode)
        return false;

    if (optimum_time <= 0)
        return true;

    return elapsed() >= optimum_time;
}

bool TimeManager::hard_stop() const {
    if (infinite_mode)
        return false;

    if (maximum_time <= 0)
        return true;

    return elapsed() >= maximum_time;
}


uint64_t TimeManager::poll_interval_mask() const {
    if (infinite_mode)
        return 4095ULL;

    const int budget = optimum_time > 0 ? optimum_time : maximum_time;

    if (budget <= 20)   return 15ULL;
    if (budget <= 50)   return 31ULL;
    if (budget <= 100)  return 63ULL;
    if (budget <= 250)  return 127ULL;
    if (budget <= 1000) return 255ULL;
    if (budget <= 5000) return 511ULL;

    return 1023ULL;
}