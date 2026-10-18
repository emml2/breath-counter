#include <cstdio>

#include "imci.hpp"

static int failures = 0;

#define CHECK(cond)                                                   \
    do {                                                              \
        if (!(cond)) {                                                \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);       \
            ++failures;                                               \
        }                                                             \
    } while (0)

int main() {
    // The threshold itself counts as fast; one breath below does not.
    CHECK(is_fast_breathing(AgeBand::UnderTwoMonths, 60.0f));
    CHECK(!is_fast_breathing(AgeBand::UnderTwoMonths, 59.0f));
    CHECK(is_fast_breathing(AgeBand::TwoToElevenMonths, 50.0f));
    CHECK(!is_fast_breathing(AgeBand::TwoToElevenMonths, 49.0f));
    CHECK(is_fast_breathing(AgeBand::OneToFiveYears, 40.0f));
    CHECK(!is_fast_breathing(AgeBand::OneToFiveYears, 39.0f));
    // Younger babies have higher thresholds.
    CHECK(fast_breathing_threshold(AgeBand::UnderTwoMonths) > fast_breathing_threshold(AgeBand::TwoToElevenMonths));
    CHECK(fast_breathing_threshold(AgeBand::TwoToElevenMonths) > fast_breathing_threshold(AgeBand::OneToFiveYears));

    std::printf(failures ? "%d failure(s)\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
