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
#include <vector>
#include "graphcut.h"

// float l1distance(const float &in1, const float &in2)
// {
//     return std::abs(in1 - in2);
// }

// float l1distance(const float *in1, const float *in2, int size)
// {
//     float ret_sum = 0.0;
//     for (int c_i = 0; c_i < size; c_i++)
//     {
//         ret_sum += abs(in1[c_i] - in2[c_i]);
//     }
//     return ret_sum;
// }

float l2distance(const float &in1, const float &in2)
{
    return std::abs(in1 - in2);
}

float l2distance(const float *in1, const float *in2, int size)
{
    float ret_sum = 0.0;
    for (int c_i = 0; c_i < size; c_i++)
    {
        ret_sum += (in1[c_i] - in2[c_i]) * (in1[c_i] - in2[c_i]);
    }
    return std::sqrt(ret_sum);
}

float l2distance(const std::vector<float> &in1, const std::vector<float> &in2)
{
    int size = in1.size();
    float ret_sum = 0.0;
    for (int c_i = 0; c_i < size; c_i++)
    {
        ret_sum += (in1[c_i] - in2[c_i]) * (in1[c_i] - in2[c_i]);
    }
    return std::sqrt(ret_sum);
}

torch::Tensor maxflow2d_cpu(const torch::Tensor &image, const torch::Tensor &prob, const float &lambda, const float &sigma, const int &connectivity)
{
    // get input dimensions
    const int batch = image.size(0);
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    std::vector<int> Xoff, Yoff;
    int offsetLen;

    if (connectivity == 0)
    {
        std::cout << "torchmaxflow: warning no connectivity provided, falling back to default 4 connectivity" << std::endl;
    }

    if ((connectivity == 4) || (connectivity == 0)) 
    {
        Xoff = {-1,  0};
        Yoff =  {0, -1};
        offsetLen = 2;
    }
    else if (connectivity == 8) 
    {
        Xoff = {-1,  0, -1};
        Yoff = { 0, -1, -1};
        offsetLen = 3;
    }
    else 
    {
        throw std::runtime_error(
            "torchmaxflow only supports 4 or 8 connectivity for 2D spatial inputs, received connectivity = " + std::to_string(connectivity) + ".");
    };
    // prepare output
    torch::Tensor label = torch::zeros({batch, 1, height, width}, image.dtype());

    // get data accessors
    auto label_ptr = label.accessor<float, 4>();
    auto image_ptr = image.accessor<float, 4>();
    auto prob_ptr = prob.accessor<float, 4>();

    // prepare graph
    // initialise with graph(num of nodes, num of edges)
    GCGraph<float> g(height * width, 2 * height * width);

    float pval, qval, l2dis, n_weight, s_weight, t_weight, prob_bg, prob_fg;
    int pIndex, qIndex;
    std::vector<float> pval_v(channel);
    std::vector<float> qval_v(channel);

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            pIndex = g.addVtx();

            // avoid log(0)
            prob_bg = std::max(prob_ptr[0][0][h][w], std::numeric_limits<float>::epsilon());
            prob_fg = std::max(prob_ptr[0][1][h][w], std::numeric_limits<float>::epsilon());
            s_weight = -log(prob_bg);
            t_weight = -log(prob_fg);

            g.addTermWeights(pIndex, s_weight, t_weight);
        }
    }

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            if (channel == 1)
            {
                pval = image_ptr[0][0][h][w];
            }
            else
            {
                for (int c_i = 0; c_i < channel; c_i++)
                {
                    pval_v[c_i] = image_ptr[0][c_i][h][w];
                }
            }

            pIndex = h * width + w;
            for (int i = 0; i < offsetLen; i++)
            {
                const int hn = h + Xoff[i];
                const int wn = w + Yoff[i];
                
                if (hn < 0 || wn < 0)
                {
                    continue;
                }

                if (channel == 1)
                {
                    qval = image_ptr[0][0][hn][wn];
                    l2dis = l2distance(pval, qval);
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        qval_v[c_i] = image_ptr[0][c_i][hn][wn];
                    }
                    l2dis = l2distance(pval_v, qval_v);
                }
                l2dis /= sqrt(Xoff[i] * Xoff[i] + Yoff[i] * Yoff[i]);
                n_weight = lambda * exp(-(l2dis * l2dis) / (2 * sigma * sigma));
                qIndex = hn * width + wn;
                g.addEdges(qIndex, pIndex, n_weight, n_weight);
            }
        }
    }

    g.maxFlow();
    // float flow = g.maxFlow();
    // std::cout << "max flow: " << flow << std::endl;

    int idx = 0;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            label_ptr[0][0][h][w] = g.inSourceSegment(idx) ? 1.0 : 0.0;
            idx++;
        }
    }
    return label;
}

torch::Tensor maxflow3d_cpu(const torch::Tensor &image, const torch::Tensor &prob, const float &lambda, const float &sigma, const int &connectivity)
{
    // get input dimensions
    const int batch = image.size(0);
    const int channel = image.size(1);
    const int depth = image.size(2);
    const int height = image.size(3);
    const int width = image.size(4);

    std::vector<int> Xoff, Yoff, Zoff;
    int offsetLen;
        
    if(connectivity == 0)
    {
        // no connectivity provided, issue warning and use default connectivity
        std::cout << "torchmaxflow: warning no connectivity provided, falling back to default 6 connectivity" << std::endl;
    }

    if ((connectivity == 6) || (connectivity == 0)) 
    {
        Xoff = {-1,  0,  0};
        Yoff = { 0, -1,  0};
        Zoff = { 0,  0, -1};
        offsetLen = 3;
    }
    else if (connectivity == 18) 
    {
        Xoff = {-1,  0,  0, -1, -1,  0};
        Yoff = { 0, -1,  0, -1,  0, -1};
        Zoff = { 0,  0, -1,  0, -1, -1};
        offsetLen = 6;
    }
    else if (connectivity == 26) 
    {
        Xoff = {-1,  0,  0, -1, -1,  0, -1};
        Yoff = { 0, -1,  0, -1,  0, -1, -1};
        Zoff = { 0,  0, -1,  0, -1, -1, -1};
        offsetLen = 7;
    }
    else 
    {
        throw std::runtime_error(
            "torchmaxflow only supports 6, 18 or 26 connectivity for 3D spatial inputs, received connectivity = " + std::to_string(connectivity) + ".");
    };

    // prepare output
    torch::Tensor label = torch::zeros({batch, 1, depth, height, width}, image.dtype());

    // get data accessors
    auto label_ptr = label.accessor<float, 5>();
    auto image_ptr = image.accessor<float, 5>();
    auto prob_ptr = prob.accessor<float, 5>();

    // prepare graph
    // initialise with graph(num of nodes, num of edges)
    GCGraph<float> g(depth * height * width, 2 * depth * height * width);

    float pval, qval, l2dis, n_weight, s_weight, t_weight, prob_bg, prob_fg;
    int pIndex, qIndex;
    std::vector<float> pval_v(channel);
    std::vector<float> qval_v(channel);
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                pIndex = g.addVtx();

                // avoid log(0)
                prob_bg = std::max(prob_ptr[0][0][d][h][w], std::numeric_limits<float>::epsilon());
                prob_fg = std::max(prob_ptr[0][1][d][h][w], std::numeric_limits<float>::epsilon());
                s_weight = -log(prob_bg);
                t_weight = -log(prob_fg);

                g.addTermWeights(pIndex, s_weight, t_weight);
            }
        }
    }

    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                if (channel == 1)
                {
                    pval = image_ptr[0][0][d][h][w];
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        pval_v[c_i] = image_ptr[0][c_i][d][h][w];
                    }
                }

                pIndex = d * height * width + h * width + w;
                for (int i = 0; i < offsetLen; i++)
                {
                    const int dn = d + Xoff[i];
                    const int hn = h + Yoff[i];
                    const int wn = w + Zoff[i];

                    if (dn < 0 || hn < 0 || wn < 0)
                    {
                        continue;
                    }

                    if (channel == 1)
                    {
                        qval = image_ptr[0][0][dn][hn][wn];
                        l2dis = l2distance(pval, qval);
                    }
                    else
                    {
                        for (int c_i = 0; c_i < channel; c_i++)
                        {
                            qval_v[c_i] = image_ptr[0][c_i][dn][hn][wn];
                        }
                        l2dis = l2distance(pval_v, qval_v); 
                    }
                    l2dis /= sqrt(Xoff[i] * Xoff[i] + Yoff[i] * Yoff[i] + Zoff[i] * Zoff[i]);
                    n_weight = lambda * exp(-(l2dis * l2dis) / (2 * sigma * sigma));
                    qIndex = dn * height * width + hn * width + wn;
                    g.addEdges(qIndex, pIndex, n_weight, n_weight);
                }
            }
        }
    }

    g.maxFlow();
    // float flow = g.maxFlow();
    // std::cout << "max flow: " << flow << std::endl;

    int idx = 0;
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                label_ptr[0][0][d][h][w] = g.inSourceSegment(idx) ? 1.0 : 0.0;
                idx++;
            }
        }
    }
    return label;
}

void add_interactive_seeds(torch::Tensor &prob, const torch::Tensor &seed, const int &num_dims)
{
    // implements Equation 7 from:
    //  Wang, Guotai, et al.
    //  "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    //  IEEE TMI (2018).

    if (num_dims == 4) // 2D
    {
        const int height = prob.size(2);
        const int width = prob.size(3);

        auto prob_ptr = prob.accessor<float, 4>();
        auto seed_ptr = seed.accessor<float, 4>();

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                if (seed_ptr[0][0][h][w] > 0)
                {
                    prob_ptr[0][0][h][w] = 1.0;
                    prob_ptr[0][1][h][w] = 0.0;
                }
                else if (seed_ptr[0][1][h][w] > 0)
                {
                    prob_ptr[0][0][h][w] = 0.0;
                    prob_ptr[0][1][h][w] = 1.0;
                }
                else
                {
                    continue;
                }
            }
        }
    }
    else if (num_dims == 5) // 3D
    {
        const int depth = prob.size(2);
        const int height = prob.size(3);
        const int width = prob.size(4);

        auto prob_ptr = prob.accessor<float, 5>();
        auto seed_ptr = seed.accessor<float, 5>();

        for (int d = 0; d < depth; d++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    if (seed_ptr[0][0][d][h][w] > 0)
                    {
                        prob_ptr[0][0][d][h][w] = 1.0;
                        prob_ptr[0][1][d][h][w] = 0.0;
                    }
                    else if (seed_ptr[0][1][d][h][w] > 0)
                    {
                        prob_ptr[0][0][d][h][w] = 0.0;
                        prob_ptr[0][1][d][h][w] = 1.0;
                    }
                    else
                    {
                        continue;
                    }
                }
            }
        }
    }
}