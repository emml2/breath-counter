#pragma once
#include <torch/torch.h>

#include "imci.hpp"

// Synthetic chest-movement signals: a noisy, slightly irregular breathing wave.
constexpr int kSampleRateHz = 10;
constexpr int kWindowSeconds = 30;
constexpr int kWindowSamples = kSampleRateHz * kWindowSeconds;

struct Batch {
    torch::Tensor signal;  // [N, 1, kWindowSamples], zero mean, unit variance per window
    torch::Tensor bpm;     // [N] true breaths per minute
    torch::Tensor band;    // [N] AgeBand as int64
};

// Draw rates from a range that straddles all three thresholds so both classes are common.
inline Batch make_batch(int64_t n, torch::Device device) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    auto bpm = 15.0f + torch::rand({n}, opts) * 75.0f;  // 15..90 breaths per minute
    auto band = torch::randint(0, kAgeBands, {n}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    auto t = torch::arange(kWindowSamples, opts) / static_cast<float>(kSampleRateHz);  // [T]
    auto hz = (bpm / 60.0f).unsqueeze(1);                                       // [N,1]
    auto phase = torch::rand({n, 1}, opts) * 6.2831853f;

    // Rate wobbles a little over the window, as a real child's breathing does.
    auto wobble = 0.05f * hz * torch::sin(2.0f * 3.14159265f * 0.1f * t + torch::rand({n, 1}, opts) * 6.28f);
    auto wave = torch::sin(6.2831853f * (hz + wobble) * t + phase);
    wave = wave + 0.25f * torch::sin(2.0f * 6.2831853f * hz * t + torch::rand({n, 1}, opts) * 6.28f);  // harmonic
    auto drift = 0.5f * (torch::rand({n, 1}, opts) - 0.5f) * t / kWindowSeconds;                // baseline drift
    auto noise = 0.35f * torch::randn({n, kWindowSamples}, opts);
    auto x = wave + drift + noise;

    x = (x - x.mean(1, true)) / (x.std(std::vector<int64_t>{1}, /*unbiased=*/false, /*keepdim=*/true) + 1e-6f);
    return {x.unsqueeze(1), bpm, band};
}
