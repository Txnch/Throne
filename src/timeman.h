#pragma once
#include <cstdint>
#include <chrono>

class TimeManager {
public:
    TimeManager();

    void reset();
    void set_limits(bool infinite,
        int move_time,
        int timeLeft,
        int inc,
        int movesToGo,
        int fullmove);

    void start();
    int elapsed() const;

    bool should_stop(uint64_t nodes);
    bool hard_stop() const;


    void set_dynamic_scale(double scale);

    uint64_t poll_interval_mask() const;

private:
    void compute_time();

    bool infinite_mode;
    bool fixed_movetime;

    int time_left;
    int increment;
    int moves_to_go;
    int fullmove_number;

    int optimum_time;
    int maximum_time;

    int base_optimum_time;
    int base_maximum_time;

    std::chrono::steady_clock::time_point start_time;
};
