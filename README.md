# NN-RGB-Image-Segmentation
Project for Computational Aspects of Robotics Course from Columbia University's School of Engineering and Applied Science, March 2023

In this homework, I trained a neural network to perform segmentation using RGB images and corresponding ground truth instance segmentation masks obtained from a simulated camera. I then used the model and RGB-D images to estimate the pose of objects.

Pytorch (https://pytorch.org/docs/stable/nn.html) Tutorials Used:
- Data Loading -> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- Neural Network -> https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- Training a Classifier -> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Project Parts:
1. Generate training set (function compute_camera_matrix in camera.py)
2. CNN for Segmentation (dataset.py, model.py, segmentation.py)
3. Pose Estimation (icp.py)
