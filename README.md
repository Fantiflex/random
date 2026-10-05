# Building Medical Imaging Tools with Deep Learning
#### CPH 100A Project 2
#### Due Date: 7PM PST Oct 16, 2025

## Introduction

Building on the machine learning foundations from Project 1, this project focuses on developing deep learning tools for medical images. You will implement CNN architectures and localization models that form the core principles of state-of-the-art medical imaging systems. While modern clinical systems use large NNs on massive medical images, you'll master the fundamental building blocks using minature 2D histopathological images (i.e. PathMNIST). These same principles—convolutional feature extraction and spatial grounding—directly translate to real-world medical imaging applications.

## Deliverables

### Individual Submissions (each student):
- Complete implementation of neural networks in `classification_models.py` and `segmentation_models.py`
- Working training loops in `classification_train.py` and `segmentation_train.py`
- **Performance targets**: >90% classification accuracy, >95% segmentation IoU
- **Due**: Individual code submissions by 7PM PST Oct 16, 2025

### Team Submissions (one per team, teams of 5):
- Analysis of your PathMNIST classification and segmentation experiments
- **Due**: Team reports by 7PM PST Oct 16, 2025

### Use of AI in this project
- **Individual Implementation**: For your learning, we encourage you NOT to use AI to implement core CNN operations, training loops, or U-Net skip connections. Understanding these fundamentals is crucial for deep learning mastery.  You  project is meant to help you develop your deep learning intuitions, and modern LLMs can rob you of that opportunity. 
- **Encouraged AI Use**: You are encouraged to use AI for utilities, visualization code, debugging assistance, and team analysis reports. Use LLMs for report writing and performance interpretation.

Remember: the primary goal is mastering deep learning fundamentals that enable clinical AI systems, not achieving peak performance through automated solutions. The secondary goal is to learn to use LLMs to make your life easier.

### Installation and Environment

**Step 1: Install Miniconda**
Download and install Miniconda from https://docs.conda.io/en/latest/miniconda.html
- Choose Python 3.10+ for your OS
- Follow installer defaults

**Step 2: Create and activate environment**
```bash
# From the project1 directory
conda create -n cph100_project2 python=3.10 -y
conda activate cph100_project2

# Install dependencies
pip install -r requirements.txt
```

**Step 3: Verify Installation**
```bash
python check_installation.py
```


**Troubleshooting:**
- **Out of space**: Clean conda cache: `conda clean --all`
- **Still having issues**: Ask LLMs, ask your team and come to office hours.


### Code Structure Overview
- `classification_main.py`: Training orchestration (provided)
- `classification_models.py`: **TODO** - Implement your classifier neural networks, including a simple MLP classifier and CNNs
- `classification_train.py`: **TODO** - Implement training/validation loops with PyTorch
- `classification_dataset.py`: PathMNIST data loading (provided)
- `segmentation_models.py`: **TODO** - Implement segmentation models, including a linear model, a CNN, and a U-Net with skip connections
- `segmentation_train.py`: **TODO** - Implement segmentation training pipeline
- `segmentation_main.py`: Segmentation training orchestration (provided)
- `synthetic_data.py`: Synthetic box generation for segmentation (provided)

## Part 1: CNN Implementation for Medical Image Classification [25 pts]

Implement and compare different architectures for PathMNIST tissue classification:

0. **Simple Linear classifier**: Flatten input images and re-implement logistic regression in PyTorch.
1. **MLP**: Flatten input images and use a multi-layer percetron (MLP) model
2. **Your CNN**: Design and implement your own convolutional architecture. Explore different architecture layouts.

**Performance Expectations (1 epoch, 100 steps):**
- MLP Model: ~32% validation accuracy
- Your CNN: >54% validation accuracy

**Final Model Requirements:**
- Train your best model on a longer training schedule. 
- **Target**: >90% validation accuracy

**Extra-Credit Opportunity: 10pts**
With a well tuned CNN, it's possible to reach >99% validation accuracy.
Consider using your experiment dispatcher as in project 1 to organize your experiments. You're encouraged to discuss your hyper-parameters with each other and collaborate with your team on finding the best model configuration.

## Part 2: Segmentation Implementation [25 pts]

Implement and compare different architectures for segmentation on PathMNIST:

0. **Simple Linear model**: Flatten input, predict flattened output
1. **MLP**: Flatten input, predict flattened output with an MLP
2. **Simple CNN**: Encoder-decoder without skip connections  
3. **U-Net**: Full U-Net with skip connections

**Performance Expectations (1 epoch, 100 steps):**
- MLP: ~43% validation IoU
- Simple CNN: >94% validation IoU  
- U-Net: >96% validation IoU

**Final Model Requirements:**
- Train your best model on a longer training schedule. 
- **Target**: >97% validation IoU (Go as high as you can! 99% is achievable)


## Team Analysis [25 pts]

### Team Report Requirements (analyze as a group):

1. **Architecture comparison**: Compare your neural networks across both classification and segmentation tasks
2. **Performance analysis**: Which design choices most improved performance? Discuss depth/width, kernel sizes, skip connections, optimization settings
