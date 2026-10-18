#pragma once
#include <torch/torch.h>

#include "signal.hpp"

// Small 1D CNN: window of chest-movement samples -> breaths per minute (scaled by 1/100).
struct RateNetImpl : torch::nn::Module {
    RateNetImpl() {
        conv1 = register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 16, 7).padding(3)));
        conv2 = register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(16, 32, 7).padding(3)));
        conv3 = register_module("conv3", torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 5).padding(2)));
        fc1 = register_module("fc1", torch::nn::Linear(64 * (kWindowSamples / 4), 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::max_pool1d(torch::relu(conv1(x)), 2);
        x = torch::max_pool1d(torch::relu(conv2(x)), 2);
        x = torch::relu(conv3(x));
        x = torch::relu(fc1(x.flatten(1)));
        return fc2(x).squeeze(1);
    }

    torch::nn::Conv1d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
TORCH_MODULE(RateNet);

constexpr float kBpmScale = 100.0f;
