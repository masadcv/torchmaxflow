# torchmaxflow: Max-flow/Min-cut in PyTorch for 2D images and 3D volumes

Pytorch-based implementation of Max-flow/Min-cut based on the following paper:

- Boykov, Yuri, and Vladimir Kolmogorov. "An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision." IEEE transactions on pattern analysis and machine intelligence 26.9 (2004): 1124-1137.

## Citation
If you use this code in your research, then please consider citing:

 **Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. ["ECONet: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation."](https://openreview.net/pdf?id=9xtE2AgD_Cc) Medical Imaging with Deep Learning (MIDL), 2022.**

## Installation instructions
`pip install torchmaxflow`

or 


```
# Clone and install from github repo

$ git clone https://github.com/masadcv/torchmaxflow
$ cd torchmaxflow
$ pip install -r requirements.txt
$ python setup.py install
```

## Example outputs
Maxflow2d

![./figures/torchmaxflow_maxflow2d.png](https://raw.githubusercontent.com/masadcv/torchmaxflow/main/figures/torchmaxflow_maxflow2d.png)

Interactive maxflow2d

![./figures/torchmaxflow_intmaxflow2d.png](https://raw.githubusercontent.com/masadcv/torchmaxflow/main/figures/torchmaxflow_intmaxflow2d.png)


![figures/figure_torchmaxflow.png](https://raw.githubusercontent.com/masadcv/torchmaxflow/main/figures/figure_torchmaxflow.png)


## Example usage

The following demonstrates a simple example showing torchmaxflow usage:
```
image = np.asarray(Image.open('data/image2d.png').convert('L'), np.float32)
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

prob = np.asarray(Image.open('data/image2d_prob.png'), np.float32)
prob = torch.from_numpy(prob).unsqueeze(0)

lamda = 20.0
sigma = 10.0

post_proc_label = torchmaxflow.maxflow(image, prob, lamda, sigma)
```

For more usage examples see: 

**2D and 3D maxflow and interactive maxflow examples**: [`demo_maxflow.py`](https://raw.githubusercontent.com/masadcv/torchmaxflow/main/demo_maxflow.py) 
 
## References
- [OpenCV's Graphcut implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc/detail/gcgraph.hpp)
- [SimpleCRF's maxflow implementation](https://github.com/HiLab-git/SimpleCRF)

This repository depends on the code for [maxflow from latest version of OpenCV](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc/detail/gcgraph.hpp), which has been included.

<!-- BibTeX:
```
@inproceedings{
asad2022econet,
title={{ECON}et: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation},
author={Muhammad Asad and Lucas Fidon and Tom Vercauteren},
booktitle={Medical Imaging with Deep Learning},
year={2022},
url={https://openreview.net/forum?id=9xtE2AgD_Cc}
}
``` -->
