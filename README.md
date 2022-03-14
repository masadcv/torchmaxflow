# torchmaxflow: Max-flow/Min-cut in PyTorch for 2D images and 3D volumes
Pytorch-based implementation of Max-flow/Min-cut based on the following paper:

- Boykov, Yuri, and Vladimir Kolmogorov. "An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision." IEEE transactions on pattern analysis and machine intelligence 26.9 (2004): 1124-1137.

This repository depends on the code for maxflow from: [https://pub.ist.ac.at/~vnk/software/maxflow-v3.04.src.zip](https://pub.ist.ac.at/~vnk/software/maxflow-v3.04.src.zip), which has been included.

## Installation instructions

`pip install git+https://github.com/masadcv/torchmaxflow`

TODO:

`pip install torchmaxflow`

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

**2D and 3D maxflow and interactive maxflow examples**: [`demo_maxflow.py`](./demo_maxflow.py) 
 
## References
- SimpleCRF's maxflow implementation: [https://github.com/HiLab-git/SimpleCRF](https://github.com/HiLab-git/SimpleCRF)
- Yuri Boykov and Vladimir Kolmogorov's maxflow implementation: [https://pub.ist.ac.at/~vnk/software/maxflow-v3.04.src.zip](https://pub.ist.ac.at/~vnk/software/maxflow-v3.04.src.zip)
## Citation
If you use this code in your research, then please consider citing:

> Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. 
>"ECONet: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation." 
>arXiv preprint arXiv:2201.04584 (2022).

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