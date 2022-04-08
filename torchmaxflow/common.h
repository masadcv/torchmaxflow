// Copyright (c) 2022, Muhammad Asad (masadcv@gmail.com)
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

void check_spatial_shape_match(const torch::Tensor &in1, const torch::Tensor &in2, const int &dims)
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
        throw std::runtime_error("torchmaxflow currently does not support CUDA, please pass CPU tensors as mytensor.cpu().");
    }
}

void check_single_batch(const torch::Tensor &in)
{
    if (in.size(0) != 1)
    {
        throw std::runtime_error("torchmaxflow currently only supports single batch input.");
    }
}

void check_binary_channels(const torch::Tensor &in)
{
    if (in.size(1) != 2)
    {
        throw std::runtime_error("torchmaxflow currently only supports binary probability.");
    }
}

void check_input_maxflow(const torch::Tensor &image, const torch::Tensor &prob, const int &num_dims)
{
    // check tensors cpu
    check_cpu(image);
    check_cpu(prob);
    
    // check batch==1
    check_single_batch(image);
    check_single_batch(prob);

    // check channels==2 for prob
    check_binary_channels(prob);

    // check spatial shapes match
    check_spatial_shape_match(image, prob, num_dims-2);    
}

void check_input_maxflow_interactive(const torch::Tensor &image, const torch::Tensor &prob, const torch::Tensor &seed, const int &num_dims)
{
    // check tensor cpu
    check_cpu(image);
    check_cpu(prob);
    check_cpu(seed);
    
    // check batch==1
    check_single_batch(image);
    check_single_batch(prob);
    check_single_batch(seed);

    // check channels==2 for prob and seeds
    check_binary_channels(prob);
    check_binary_channels(seed);

    // check spatial shapes match
    check_spatial_shape_match(image, prob, num_dims-2);    
    check_spatial_shape_match(image, seed, num_dims-2);    
}