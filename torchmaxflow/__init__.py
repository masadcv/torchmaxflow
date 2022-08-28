# Copyright (c) 2022, Muhammad Asad (masadcv@gmail.com)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Copyright (c) 2022, Muhammad Asad (masadcv@gmail.com)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torchmaxflowcpp


def maxflow(
    image: torch.Tensor, 
    prob: torch.Tensor, 
    lamda: float, 
    sigma: float, 
    connectivity: int = 0
) -> torch.Tensor:
    r"""Computes Max-flow/Min-cut in PyTorch for 2D images and 3D volumes follwing methods described in:
    
    Boykov, Yuri, and Vladimir Kolmogorov. 
    "An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision." 
    IEEE transactions on pattern analysis and machine intelligence 26.9 (2004): 1124-1137.
    
    The function expects input as torch.Tensor, which can be a 2D image or 3D volume
    Args:
        image: input image, can be 2D [C, H, W] or 3D [C, D, H, W].
        prob: probability in range [0, 1] and dimensions [2, H, W] or [2, D, H, W].
        lamda: weighting factor for establishing relationship between unary and pairwise terms.
        sigma: standard deviation of intensity values in neighbourhood.
        connectivity: connectivity to use, can be from [4, 8] for 2D and [6, 18, 24] for 3D
    Returns:
        torch.Tensor with maxflow output
    """
    return torchmaxflowcpp.maxflow(
        image, prob, lamda, sigma, connectivity
    )

def maxflow_interactive(
    image: torch.Tensor, 
    prob: torch.Tensor, 
    seed: torch.Tensor,
    lamda: float, 
    sigma: float, 
    connectivity: int = 0
) -> torch.Tensor:
    r"""Computes Interactive Max-flow/Min-cut in PyTorch for 2D images and 3D volumes follwing methods described in:
    
    Boykov, Yuri, and Vladimir Kolmogorov. 
    "An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision." 
    IEEE transactions on pattern analysis and machine intelligence 26.9 (2004): 1124-1137.
    
    The function expects input as torch.Tensor, which can be a 2D image or 3D volume
    Args:
        image: input image, can be 2D [C, H, W] or 3D [C, D, H, W].
        prob: probability in range [0, 1] and dimensions [2, H, W] or [2, D, H, W].
        seed: seed providing user interaction input with dimensions [2, H, W] or [2, D, H, W].
        lamda: weighting factor for establishing relationship between unary and pairwise terms.
        sigma: standard deviation of intensity values in neighbourhood.
        connectivity: connectivity to use, can be from [4, 8] for 2D and [6, 18, 24] for 3D
    Returns:
        torch.Tensor with maxflow output
    """
    return torchmaxflowcpp.maxflow_interactive(
        image, prob, seed, lamda, sigma, connectivity
    )