# ML---Digit-Detection
This project focuses on developing a machine learning model to identify numbers within images, laying the groundwork for future text detection algorithms.
Number Identification for Text Detection
This project focuses on developing a machine learning model to identify numbers within images, laying the groundwork for future text detection algorithms.

## **Overview**
This project aims to classify and localize numerical digits in images, preparing for more complex text detection tasks. It involves preprocessing, training a convolutional neural network (CNN) model using TensorFlow/Keras, and evaluating its performance.

## **Dataset**
The dataset used consists of images containing single digits. It is divided into training, validation, and test sets with appropriate preprocessing applied.

## **Model Architecture**
The CNN model architecture includes convolutional layers, max pooling, dropout for regularization, and fully connected layers with ReLU and softmax activations for classification.

## **Training**
The model is trained using Adam optimizer with categorical crossentropy loss. Data augmentation techniques such as image shifting, zooming, and rotation are applied to enhance robustness.

## **Evaluation**
Model performance is evaluated on a separate test set using accuracy metrics. Training and validation loss/accuracy curves are plotted to assess training progress.

## **Dependencies**
 * **Python 3.x:** Programming language used for development.

* **TensorFlow 2.x:** Open-source machine learning framework by Google.

Build and train deep learning models, manage computational graphs, and optimize GPU computation.
* **OpenCV:** Library for real-time computer vision tasks.

Image and video processing, including reading, resizing, manipulation, and object detection.
* **NumPy:** Fundamental package for numerical computing in Python.

Support for multi-dimensional arrays, matrices, and mathematical operations, essential for data manipulation.
* **Matplotlib:** Plotting library for Python.

Create visualizations such as graphs for training/validation metrics and display sample images.
* **scikit-learn:** Machine learning library for Python.

Tools for data mining, evaluation of model performance, and potentially preprocessing and feature extraction.


## **Usage**
Training: Run python train.py to train the model using the prepared dataset.
Prediction: Run python predict.py to use the trained model for real-time prediction using webcam capture.

## **Credits**
Author: Afsheen Wasfiya Abdul Wahab
LinkedIn: https://www.linkedin.com/in/afsheen-wasfiya-abdul-wahab-33a504257/
