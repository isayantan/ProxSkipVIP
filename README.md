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
   * [Generate Games](#generate-games)
   * [Strongly-monotone Quadratic Games](#strongly-monotone-quadratic-games)
   * [Robust Least Square](#robust-least-square)
   * [Fine-Tuned Stepsize](#Fine-Tuned-Stepsize)
   * [ProxSkip-SGDA-FL vs ProxSkip-L-SVRGDA-FL](#ProxSkip-SGDA-FL-vs-ProxSkip-L-SVRGDA-FL)
   * [Performance on Data with Varying Heterogeneity](Performance-on-Data-with-Varying-Heterogeneity)
   
<!--te-->

## Requirements

The required Python packages for running the files are listed below
 * ```numpy```
 * ```matplotlib```
 * ```math```
 * ```time```
 * ```pandas```
 * ```sklearn.datasets```
 * ```seaborn```
 * ```scipy.optimize```

## Generate Games
If you want to test the different optimizers on your own game, use ```QuadGame()``` to generate a quadratic game and ```MatrixGame()``` to generate a matrix game from [model.py](model.py). You need to use the ```grad``` method to return oracles evaluated at a particular value for the game.

## Strongly-monotone Quadratic Games

In figure 1 of our work we compare the performance of ProxSkip-GDA-FL (ProxSkip-SGDA-FL) with Local GDA (Local SGDA) and Local EG (Local SEG) on strongly-monotone quadratic games. Please the run the code in [QuadxDet.py](QuadxDet.py) and [QuadxStoch.py](QuadxStoch1.py) for deterministic and stochastic setting respectively. 

## Robust Least Square

In figures 2 and 3 of our work we compare the performance of ProxSkip-GDA-FL (ProxSkip-SGDA-FL) with Local GDA (Local SGDA) and Local EG (Local SEG) for Robust Least Square problems. Please the run the code in [RLS_Dataset2.ipynb](RLS_Dataset2.ipynb) and [RLS_Dataset1.ipynb](RLS_Dataset1.ipynb) to reproduce the plots of figures 2 and 3 respectively.

## Fine-Tuned Stepsize

In figure 4, we compare the performance of ProxSkip-GDA-FL (ProxSkip-SGDA-FL) with Local GDA (Local SGDA) and Local EG (Local SEG) on the strongly monotone quadratic game using tuned stepsizes. To reproduce the plots of figure 4, please run the notebook [ProxSkip_tuned_HeterogeneousData.ipynb](ProxSkip_tuned_HeterogeneousData.ipynb). 

## ProxSkip-SGDA-FL vs ProxSkip-L-SVRGDA-FL

In figure 5, we compare ProxSkip-SGDA-FL with ProxSkip-L-SVRGDA-FL for tuned and theoretical step-sizes. Please run the notebooks [ProxSkip SVRGDA vs ProxSkip with tuned stepsizes.ipynb](https://github.com/isayantan/ProxSkipVIP/blob/main/ProxSkip%20SVRGDA%20vs%20ProxSkip%20with%20tuned%20stepsizes.ipynb) and [ProxSkip vs ProxSkip L SVRGDA Theoretical Stepsize.ipynb](https://github.com/isayantan/ProxSkipVIP/blob/main/ProxSkip%20vs%20ProxSkip%20L%20SVRGDA%20Theoretical%20Stepsize.ipynb)to reproduce the plots of figure 5.


## Performance on Data with Varying Heterogeneity
In figure 7, we compare the perfromance of ProxSkip-GDA-FL, Local GDA and Local EG on data with varying heterogenity. To reproduce the plots of figure 7, please run the notebook [ProxSkip_Varying_Heterogenety.ipynb](ProxSkip_Varying_Heterogenety.ipynb).
