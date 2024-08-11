# Semantic segmentation using deep learning

The project dealt with the implementation of semantic segmentation of images using deep learning . The goal of the project was to implement a neural network that performs the segmentation, to write Python code that performs the task and to improve performance by changing hyperparameters or other things (such as adding a spatial LSTM layer as we did).

![image](https://github.com/user-attachments/assets/b9e1a2df-5f47-4a16-82a0-bb1959314962)

After trials and improvements, the networks chosen for examination and optimization were Resnet50, EfficientNetB4 and EfficientNetV2. After passing through the network, the output (before classification) was scanned with zig-zag scan, and entered a spatial LSTM layer, which utilizes spatial relationships between neighboring pixels for the purpose of improving accuracy.

![image](https://github.com/user-attachments/assets/dba0d83a-c67d-4ab2-89fe-263a8ed99423)

We received that the addition of the LSTM layer in each network created a significant improvement in the test accuracy percentage. We received a 1.5% improvement of the accuracy percentage on the test set in each network.
In addition, the most successful network was ResNet50 neural network, with an accuracy of 93.04% on the test set after the LSTM layer.
Adding the extended LSTM layer turned out to be successful, and can be used to improve the accuracy in segmentation tasks with purpose of improving the overall accuracy on all labels.

![image](https://github.com/user-attachments/assets/2e9438fd-cd75-46bd-87ab-ce4aa21c1b91)

# What inside?

We have 2 folders:
- python: all python files, including main.py.
- plots: all plots, for 3 examined networks, including accuracy & loss curves, confusion matrixes segmented images and aditional analysys.

# How to run?
1. In the beginning of main.py, chnage the necessary variables according to desired settings.
   Including:
   Choise of NN - ResNet50, EfficientNetv2, EfficientNetB4
   Cohise if train the model, if test it, or both.
2. Run the main.py file code using python.
