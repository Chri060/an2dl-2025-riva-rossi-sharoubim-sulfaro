# Image Classification And Segmentation With Deep Learning

<div align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python"> 
    <img src="https://img.shields.io/badge/Tensorflow-5C3EE8?logo=tensorflow&logoColor=white&style=for-the-badge" alt="Tensorflow Lite">
    <img src="https://img.shields.io/badge/Scikit_learn-FF6F00?logo=scikitlearn&logoColor=white&style=for-the-badge" alt="Scikit">
</div>


This repository contains two deep learning projects: one focused on blood cell image classification, and the other on Mars terrain segmentation: 
- [_Image classification_](/first_challenge): the first project tackles the challenge of classifying 96×96 RGB images of blood cells into eight categories, each representing a distinct cell state. The goal is to build a model that not only achieves high accuracy but also generalizes effectively across the dataset, ensuring reliable identification of the correct cell type from unseen images.
- [_Image segmentation_](/second_challenge): the second project shifts focus to semantic segmentation, where the task involves analyzing 64×128 gray-scale images of Martian terrain. Using a dataset of labeled samples and unlabeled test samples, the objective is to correctly segment the terrain into five distinct classes. The work emphasizes maximizing mean Intersection over Union by experimenting with and comparing a range of Deep Learning approaches

## Results

For the blood cell classification project, the final model achieved an accuracy of 86%, demonstrating strong performance in correctly identifying the different cell states across all categories.

For the Mars terrain segmentation project, the model reached a mean Intersection over Union (mIoU) of 48%, reflecting a reasonable segmentation performance given the complexity of the five-class problem.
## Installation

Both projects are implemented as Python notebooks and can be run either locally (using conda or a standard Python environment) or directly on Google Colab.

To set up the environment locally, clone the repository:
```bash
git clone https://github.com/Chri060/an2dl-2025-riva-rossi-sharoubim-sulfaro
cd your-repo
```
If you prefer using conda, you can create and activate a new environment:
```bash
conda create -n dl-projects python=3.9
conda activate dl-projects
```
Tenm you need to install all required dependencies.
## Authors

- [Filippo Riva](https://github.com/FilippoRiva)
- [Christian Rossi](https://github.com/Chri060)
- [Carlo Sharoubim](https://github.com/kirolosharoubim)
- [Antonio Sulfaro](https://github.com/AntonioSulfaro)

