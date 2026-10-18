#include <cstdio>

#include <torch/torch.h>

#include "imci.hpp"
#include "model.hpp"
#include "signal.hpp"

namespace {

struct Metrics {
    double mae_bpm;
    double fast_accuracy;
    double fast_sensitivity;  // of truly fast breathers, how many were flagged
    double fast_specificity;
};

Metrics evaluate(RateNet& net, torch::Device device, int64_t n) {
    torch::NoGradGuard no_grad;
    net->eval();
    auto b = make_batch(n, device);
    auto pred = net->forward(b.signal) * kBpmScale;

    auto pred_c = pred.to(torch::kCPU);
    auto true_c = b.bpm.to(torch::kCPU);
    auto band_c = b.band.to(torch::kCPU);

    int64_t tp = 0, tn = 0, fp = 0, fn = 0;
    for (int64_t i = 0; i < n; ++i) {
        auto band = static_cast<AgeBand>(band_c[i].item<int64_t>());
        bool truth = is_fast_breathing(band, true_c[i].item<float>());
        bool call = is_fast_breathing(band, pred_c[i].item<float>());
        if (truth && call) ++tp; else if (truth) ++fn; else if (call) ++fp; else ++tn;
    }
    return {(pred_c - true_c).abs().mean().item<double>(),
            static_cast<double>(tp + tn) / static_cast<double>(n),
            tp + fn ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0,
            tn + fp ? static_cast<double>(tn) / static_cast<double>(tn + fp) : 0.0};
}

}  // namespace

int main() {
    torch::manual_seed(7);
    torch::Device device(torch::kCPU);

    RateNet net;
    net->to(device);
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(1e-3));

    constexpr int kSteps = 1500;
    constexpr int kBatch = 64;

    std::printf("breath-counter: LibTorch %d.%d  (UNREVIEWED PROTOTYPE, not for clinical use)\n",
                TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR);

    for (int step = 1; step <= kSteps; ++step) {
        net->train();
        auto b = make_batch(kBatch, device);
        opt.zero_grad();
        auto loss = torch::smooth_l1_loss(net->forward(b.signal), b.bpm / kBpmScale);
        loss.backward();
        opt.step();

        if (step % 250 == 0) {
            std::printf("step %4d  loss %.5f\n", step, loss.item<float>());
        }
    }

    auto m = evaluate(net, device, 4000);
    std::printf("\nheld-out synthetic windows (n=4000)\n");
    std::printf("  mean abs error     %.2f breaths/min\n", m.mae_bpm);
    std::printf("  fast-breathing acc %.1f%%\n", m.fast_accuracy * 100.0);
    std::printf("  sensitivity        %.1f%%  (fast breathers correctly flagged)\n", m.fast_sensitivity * 100.0);
    std::printf("  specificity        %.1f%%\n", m.fast_specificity * 100.0);

    torch::save(net, "rate_net.pt");
    std::printf("\nsaved rate_net.pt\n");
    return 0;
}
