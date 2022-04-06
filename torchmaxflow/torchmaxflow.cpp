#include <torch/extension.h>
#include <vector>
#include "torchmaxflow.h"
#include "common.h"

void check_shape_match(const torch::Tensor &in1, const torch::Tensor &in2, const int &dims)
{
    for(int i=0; i < dims; i++)
    {
        if(in1.size(2+i) != in2.size(2+i))
        {
            std::cout << "Tensor1 ";
            print_shape(in1);
            std::cout << "Tensor2 ";
            print_shape(in2);
            AT_ERROR("Error: shapes of input tensors do not match");
        }
    }
}

torch::Tensor maxflow(const torch::Tensor &image, const torch::Tensor &prob, const float &lambda, const float &sigma)
{
    // check input dimensions
    // could be 2D or 3D tensors of shapes
    // 2D: 1 x C x H x W  (4 dims)
    // 3D: 1 x C x D x H x W (5 dims)
    const int num_dims = prob.dim();
    if (num_dims != 4 && num_dims != 5)
    {
        throw std::runtime_error(
            "function only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2));
    }

    if (image.is_cuda() || prob.is_cuda())
    {
        AT_ERROR("Library currently does not support CUDA, please pass CPU tensors as mytensor.cpu().");
    }

    if (image.size(0) != 1 || prob.size(0) != 1)
    {
        AT_ERROR("Library currently only supports single batch input.");
    }

    if (prob.size(1) != 2)
    {
        AT_ERROR("Library currently only supports binary probability.");
    }

    // 2D case: 1 x C x H x W
    if (num_dims == 4)
    {
        check_shape_match(image, prob, 2);
        return maxflow2d_cpu(image, prob, lambda, sigma);
    }

    // 3D case: 1 x C x D x H x W
    else if (num_dims == 5)
    {
        check_shape_match(image, prob, 3);
        return maxflow3d_cpu(image, prob, lambda, sigma);
    }
}

torch::Tensor maxflow_interactive(const torch::Tensor &image, torch::Tensor &prob, const torch::Tensor &seed, const float &lambda, const float &sigma)
{
    // check input dimensions
    // could be 2D or 3D tensors of shapes
    // 2D: 1 x C x H x W  (4 dims)
    // 3D: 1 x C x D x H x W (5 dims)
    const int num_dims = prob.dim();
    if (num_dims != 4 && num_dims != 5)
    {
        throw std::runtime_error(
            "function only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2));
    }

    if (image.is_cuda() || prob.is_cuda() || seed.is_cuda())
    {
        AT_ERROR("Library currently does not support CUDA, please pass CPU tensors as mytensor.cpu().");
    }

    if (image.size(0) != 1 || prob.size(0) != 1 || seed.size(0) != 1)
    {
        AT_ERROR("Library currently only supports single batch input.");
    }

    if (prob.size(1) != 2)
    {
        AT_ERROR("Library currently only supports binary probability.");
    }

    if (seed.size(1) != 2)
    {
        AT_ERROR("Library currently only supports binary seeds.");
    }

    // add interactive points to prob using seed locations
    add_interactive_seeds(prob, seed, num_dims);

    // 2D case: 1 x C x H x W
    if (num_dims == 4)
    {
        check_shape_match(image, prob, 2);
        check_shape_match(image, seed, 2);
        return maxflow2d_cpu(image, prob, lambda, sigma);
    }
    // 3D case: 1 x C x D x H x W
    else
    {
        check_shape_match(image, prob, 3);
        check_shape_match(image, seed, 3);
        return maxflow3d_cpu(image, prob, lambda, sigma);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("maxflow", &maxflow, "Max-flow min-cut inference for 2D/3D tensors");
    m.def("maxflow_interactive", &maxflow_interactive, "Max-flow min-cut inference for 2D/3D tensors with interactive input");
}