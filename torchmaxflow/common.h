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

void check_shape_match(const torch::Tensor &in1, const torch::Tensor &in2, const int &dims)
{
    if (in1.dim() != in2.dim())
    {
        throw std::runtime_error("dimensions of input tensors do not match " 
            + std::to_string(in1.dim() - 2) + " vs " + std::to_string(in2.dim() - 2));
    }

    for(int i=0; i < dims; i++)
    {
        if(in1.size(2+i) != in2.size(2+i))
        {
            std::cout << "Tensor1 ";
            print_shape(in1);
            std::cout << "Tensor2 ";
            print_shape(in2);
            throw std::runtime_error("shapes of input tensors do not match");
        }
    }
}

void check_cpu(const torch::Tensor &in)
{
    if (in.is_cuda())
    {
        throw std::runtime_error("Library currently does not support CUDA, please pass CPU tensors as mytensor.cpu().");
    }
}

void check_single_batch(const torch::Tensor &in)
{
    if (in.size(0) != 1)
    {
        throw std::runtime_error("Library currently only supports single batch input.");
    }
}

void check_binary_channels(const torch::Tensor &in)
{
    if (in.size(1) != 2)
    {
        throw std::runtime_error("Library currently only supports binary probability.");
    }
}

void check_input_maxflow(const torch::Tensor &image, const torch::Tensor &prob, const int &num_dims)
{
    // check input dimensions
    check_cpu(image);
    check_cpu(prob);
    
    check_single_batch(image);
    check_single_batch(prob);

    check_binary_channels(prob);

    check_shape_match(image, prob, num_dims-2);    
}

void check_input_maxflow_interactive(const torch::Tensor &image, const torch::Tensor &prob, const torch::Tensor &seed, const int &num_dims)
{
    // check input dimensions
    check_cpu(image);
    check_cpu(prob);
    check_cpu(seed);
    
    check_single_batch(image);
    check_single_batch(prob);
    check_single_batch(seed);

    check_binary_channels(prob);
    check_binary_channels(seed);

    check_shape_match(image, prob, num_dims-2);    
    check_shape_match(image, seed, num_dims-2);    
}