# Communication-Efficient Gradient Descent-Accent Methods for Distributed Variational Inequalities: Unified Analysis and Local Updates

This repository documents the code to reproduce the experiments reported in the paper:

> [Communication-Efficient Gradient Descent-Accent Methods for Distributed Variational Inequalities: Unified Analysis and Local Updates](https://arxiv.org/pdf/2306.05100.pdf)

In this paper, we provide a unified convergence analysis of communication-efficient local training methods for distributed variational inequality problems (VIPs). Our approach is based on a general key assumption on the stochastic estimates that allows us to propose and analyze several novel local training algorithms under a single framework for solving a class of structured non-monotone VIPs. We present the first local gradient descent-accent algorithms with provable _improved communication_ complexity for solving distributed variational inequalities on heterogeneous data. Here is a screenshot of the algorithm for VIP:

![ProxSkipVIPFLalgo](image/ProxSkipVIPFLalgo.png)

If you find our code useful, please cite our work as follow:

```
@article{zhang2023communication,
  title={Communication-Efficient Gradient Descent-Accent Methods for Distributed Variational Inequalities: Unified Analysis and Local Updates},
  author={Zhang, Siqi and Choudhury, Sayantan and Stich, Sebastian U and Loizou, Nicolas},
  journal={arXiv preprint arXiv:2306.05100},
  year={2023}
}
```


## Table of Contents

<!--ts-->
   * [Requirements](#requirements)
   * [Strongly-monotone Quadratic Games](#strongly-monotone-quadratic-games)
   * 
   
<!--te-->

## Requirements

The required Python packages for running the files are listed below
 * ```numpy```
 * ```matplotlib```
 * ```math```

## Strongly-monotone Quadratic Games


