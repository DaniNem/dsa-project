# Exploring Dimensionality Reduction and Loss Functions for Image Classification in Resource-Constrained Environments

This repo contains the source code for the submission of the final project in the Data Streaming Algorithms in Reichman University, February 2023.

## Introduction

Our paper discusses the importance of dimensionality reduction in machine learning, particularly for handling large datasets with limited resources. Various methods for dimensionality reduction, including PCA, t-SNE, and autoencoders, each have their own advantages and limitations. The use of dimensionality reduction in edge IoT devices is particularly important due to their limited processing power, memory, and battery life. Our paper investigates the effectiveness of two loss functions, central loss and supervised contrastive loss, for image classification and explores the effect of dimensionality reduction on learned features using sparse random projection, PCA, and trained auto-encoders. The k-Nearest Neighbor Classification algorithm is used to evaluate the quality of learned features in the reduced dimensionality space, and the confidence threshold of the KNN classifier is analyzed for different dimensionality reductions.

## Credits

In one of the experiments we used [SupContrast](https://github.com/HobbitLong/SupContrast).
