#pragma once

#include <torch/extension.h>
#include "common.h"

torch::Tensor maxflow2d_cpu(
    const torch::Tensor &image,
    const torch::Tensor &prob,
    const float &lambda,
    const float &sigma);

torch::Tensor maxflow3d_cpu(
    const torch::Tensor &image,
    const torch::Tensor &prob,
    const float &lambda,
    const float &sigma);

torch::Tensor maxflow(
    const torch::Tensor &image,
    const torch::Tensor &prob,
    const float &lambda,
    const float &sigma);

void add_interactive_seeds(
    torch::Tensor &prob, 
    const torch::Tensor &seed, 
    const int &num_dims);
