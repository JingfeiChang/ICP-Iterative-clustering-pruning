# Iterative clustering pruning (ICP)
### Code for the paper
[Iterative clustering pruning for convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0950705123001363)
<br>Chang J, Lu Y, Xue P, Xu Y, Wei Z. Iterative clustering pruning for convolutional neural networks. Knowledge-Based Systems. 2023 Apr 8;265:110386.
<br>https://doi.org/10.1016/j.knosys.2023.110386

### Introduction to Methods
Convolutional neural networks (CNNs) have shown excellent performance in numerous computer vision tasks. However, the high computational and memory demands in computer vision tasks prohibit the practical applications of CNNs on edge computing devices. Existing iterative pruning methods suffer from insufficient accuracy recovery after each pruning, which severely affects the importance evaluation of the parameters. Moreover, channel pruning based on the magnitude of parameters often results in performance loss. In this context, we propose an iterative clustering pruning method named ICP together with knowledge transfer for channels. First, channel clustering pruning is performed based on the similarity between feature maps. Then, the intermediate and output features of the original network are applied to guide the learning of the compressed network after each pruning step to quickly recover the network performance and then implement the next pruning operation. Pruning and knowledge transfer are performed alternately to achieve accurate compression of the convolutional network. Finally, we demonstrate the effectiveness of the proposed method on the CIFAR-10, CIFAR-100, and ILSVRC-2012 datasets by pruning VGGNet, ResNet, and GoogLeNet. Our pruning scheme can typically reduce parameters and Floating-point Operations (FLOPs) of the network without harming accuracy significantly. In addition, the ICP was verified to have good practical generalization by compressing the SSD network on the object detection dataset PASCAL VOC.

### The framework of ICP algorithm.
![figure1.jpg](https://github.com/JingfeiChang/ICP-Iterative-clustering-pruning/blob/main/Figures/figure1.jpg)

### Experimental results
#### ***Results on CIFAR-10***
<br>![Table1](https://github.com/JingfeiChang/ICP-Iterative-clustering-pruning/blob/main/Tables/Table1.jpg)
<br>![Table2](https://github.com/JingfeiChang/ICP-Iterative-clustering-pruning/blob/main/Tables/Table2.jpg)
---
#### ***Results on CIFAR-100***
<br>![Table3](https://github.com/JingfeiChang/ICP-Iterative-clustering-pruning/blob/main/Tables/Table3.jpg)
---
#### ***Results on ImageNet***
<br>![Table4](https://github.com/JingfeiChang/ICP-Iterative-clustering-pruning/blob/main/Tables/Table4.jpg)
---
#### ***Results on object detection***
![figure2.jpg](https://github.com/JingfeiChang/ICP-Iterative-clustering-pruning/blob/main/Figures/figure2.jpg)
