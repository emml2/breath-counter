#pragma once
// WHO IMCI fast-breathing thresholds. This is a fixed clinical rule, never learned:
// the network only estimates a breathing rate, and this table turns it into a sign.
// PROTOTYPE: values copied from the WHO IMCI chart booklet; not clinically reviewed.

enum class AgeBand { UnderTwoMonths = 0, TwoToElevenMonths = 1, OneToFiveYears = 2 };

constexpr int kAgeBands = 3;

// Breaths per minute at or above which breathing counts as "fast".
constexpr float fast_breathing_threshold(AgeBand band) {
    switch (band) {
        case AgeBand::UnderTwoMonths:     return 60.0f;
        case AgeBand::TwoToElevenMonths:  return 50.0f;
        case AgeBand::OneToFiveYears:     return 40.0f;
    }
    return 40.0f;  // unreachable; keeps -Wreturn-type quiet
}

constexpr bool is_fast_breathing(AgeBand band, float breaths_per_minute) {
    return breaths_per_minute >= fast_breathing_threshold(band);
}
