# DM-Count: Distribution Matching for Crowd Counting

This repository contains the implementation of DM-Count, a novel method for crowd counting that leverages Optimal Transport (OT) to match predicted and ground-truth density distributions. The repository is developed with the aim to reproduce the results from the paper as part of course project and agin understanding of Desnity Distribution Analysis techniques.

## Summary of the Paper

DM-Count introduces a new approach to crowd counting by avoiding Gaussian smoothing of dot annotations and using Optimal Transport (OT) and Total Variation (TV) losses. The paper shows previous state of the art models which use imposing Gaussian annotations hurt generalization perfiormance of crowd counting network. Primary reason of annotating dot into a Gaussian blob is to make ground truth more balanced making it easier to train the model but paper emphasizes that the performance is now dependant on "quality of this “pseudo ground truth”" and it is non-trivial to set right widhts of gaussian blobs specially in crowded scenes. Another approach in previous state of art results was Bayesian Loss in this Each pixel value of a smoothed ground truth density map is the posterior probability of the corresponding annotation dot given the location of that pixel. The paper points two major problems with this approach and these are First, it also requires a Gaussian kernel to construct the likelihood function for each annotated dot, which involves setting the kernel width. Second, this loss corresponds to an underdetermined system of equations with infinitely many solutions  Therefore, the predicted density map could be very different from the ground truth density map. Hence the paper propses using combination of counting loss, scaled optimal Transport Loss(optimal cost to transform one probability distribution to another) optimized using Sinkhorn algorithm and scaled Total variation Loss (for low density areas and increasing stability of training procedure) to train a deep neural network f for density map estimation by minimizing L(f)=1/K
The paper shows superior performance in both dense and sparse crowd scenarios across major datasets.

## Features

- Avoids limitations of Gaussian smoothing methods
- Uses Optimal Transport for direct density distribution comparison
- Incorporates Total Variation loss for training stability
- State-of-the-art performance on major crowd counting datasets

## Datasets

The method has been tested on the following datasets:

- UCF-QNRF
- NWPU
- ShanghaiTech (Part A and B)
- UCF-CC-50

## Results

DM-Count achieves state-of-the-art results on various datasets. For example:

- NWPU: MAE reduced from 105.4 to 88.4 (16% improvement) at that time.
- Improved PSNR and SSIM metrics compared to Pixel wise and Bayesian Loss, indicating sharper and more accurate density maps

## Implementation
Official github repo(mentioned in paper):https://github.com/cvlab-stonybrook/DM-Count
