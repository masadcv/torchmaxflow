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

#include <torch/extension.h>
#include "torchmaxflow.h"
#include "common.h"

torch::Tensor maxflow(const torch::Tensor &image, const torch::Tensor &prob, const float &lambda, const float &sigma, const int &connectivity)
{
    // could be 2D or 3D tensors of shapes
    // 2D: 1 x C x H x W  (4 dims)
    // 3D: 1 x C x D x H x W (5 dims)
    const int num_dims = prob.dim();
    check_input_maxflow(image, prob, num_dims);

    // 2D case: 1 x C x H x W
    if (num_dims == 4)
    {
        return maxflow2d_cpu(image, prob, lambda, sigma, connectivity);
    }
    // 3D case: 1 x C x D x H x W
    else if (num_dims == 5)
    {
        return maxflow3d_cpu(image, prob, lambda, sigma, connectivity);
    }
    else
    {
        throw std::runtime_error(
            "torchmaxflow only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2) + "D inputs");
    }
}

torch::Tensor maxflow_interactive(const torch::Tensor &image, torch::Tensor &prob, const torch::Tensor &seed, const float &lambda, const float &sigma, const int &connectivity)
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
        return maxflow2d_cpu(image, prob, lambda, sigma, connectivity);
    }
    // 3D case: 1 x C x D x H x W
    else if (num_dims == 5)
    {
        return maxflow3d_cpu(image, prob, lambda, sigma, connectivity);
    }
    else
    {
        throw std::runtime_error(
            "torchmaxflow only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2) + "D inputs");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("maxflow", &maxflow, "Max-flow min-cut inference for 2D/3D tensors", 
        py::arg("image"), py::arg("prob"), py::arg("lambda"), py::arg("sigma"), py::arg("connectivity")=0
    );
    m.def("maxflow_interactive", &maxflow_interactive, "Max-flow min-cut inference for 2D/3D tensors with interactive input",
        py::arg("image"), py::arg("prob"),  py::arg("seed"), py::arg("lambda"), py::arg("sigma"), py::arg("connectivity")=0
    );
}