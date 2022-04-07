#include <torch/extension.h>
#include "torchmaxflow.h"
#include "common.h"

torch::Tensor maxflow(const torch::Tensor &image, const torch::Tensor &prob, const float &lambda, const float &sigma)
{
    // could be 2D or 3D tensors of shapes
    // 2D: 1 x C x H x W  (4 dims)
    // 3D: 1 x C x D x H x W (5 dims)
    const int num_dims = prob.dim();
    check_input_maxflow(image, prob, num_dims);

    // 2D case: 1 x C x H x W
    if (num_dims == 4)
    {
        return maxflow2d_cpu(image, prob, lambda, sigma);
    }
    // 3D case: 1 x C x D x H x W
    else if (num_dims == 5)
    {
        return maxflow3d_cpu(image, prob, lambda, sigma);
    }
    else
    {
        throw std::runtime_error(
            "Library only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2) + "D inputs");
    }
}

torch::Tensor maxflow_interactive(const torch::Tensor &image, torch::Tensor &prob, const torch::Tensor &seed, const float &lambda, const float &sigma)
{
    // check input dimensions
    // could be 2D or 3D tensors of shapes
    // 2D: 1 x C x H x W  (4 dims)
    // 3D: 1 x C x D x H x W (5 dims)
    const int num_dims = prob.dim();
    check_input_maxflow_interactive(image, prob, seed, num_dims);

    // add interactive points to prob using seed locations
    add_interactive_seeds(prob, seed, num_dims);

    // 2D case: 1 x C x H x W
    if (num_dims == 4)
    {
        return maxflow2d_cpu(image, prob, lambda, sigma);
    }
    // 3D case: 1 x C x D x H x W
    else if (num_dims == 5)
    {
        return maxflow3d_cpu(image, prob, lambda, sigma);
    }
    else
    {
        throw std::runtime_error(
            "Library only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2) + "D inputs");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("maxflow", &maxflow, "Max-flow min-cut inference for 2D/3D tensors");
    m.def("maxflow_interactive", &maxflow_interactive, "Max-flow min-cut inference for 2D/3D tensors with interactive input");
}