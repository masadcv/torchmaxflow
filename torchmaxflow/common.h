#pragma once
#include <torch/extension.h>
#include <iostream>

void print_shape(const torch::Tensor &data)
{
    auto num_dims = data.dim();
    std::cout << "Shape: (";
    for (int dim = 0; dim < num_dims; dim++)
    {
        std::cout << data.size(dim);
        if (dim != num_dims - 1)
        {
            std::cout << ", ";
        }
        else
        {
            std::cout << ")" << std::endl;
        }
    }
}


// #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
// #define CHECK_CONTIGUOUS_CUDA(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)