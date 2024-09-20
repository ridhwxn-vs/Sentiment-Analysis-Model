## AI Internship Notes - Teachnook Edtech Bangalore

### Overview
This repository consists of notes and code compiled during my internship at Teachnook Edtech, Bangalore as an AI Intern. The source code will be added shortly and referenced where required.

---

### Key Concepts in Machine Learning

- **Supervised Learning**:  
  A machine learning model is trained on a labeled dataset where input and output pairs are provided.

- **Unsupervised Learning**:  
  The model is trained on an unlabeled dataset. The goal is to discover hidden patterns, structures, or relationships in the data.

- **Reinforcement Learning**:  
  An agent learns through feedback from its environment, aiming to maximize cumulative rewards.

- **Deep Learning**:  
  A branch of machine learning that involves neural networks with multiple layers, enabling models to learn complex patterns.

---

### Supervised Classification Model
- **Definition**: A model is trained using labeled data and tested using a test set. The model then classifies data into one or more classes.
  
- **Types**:
  - **Binary Classifier**: Classifies data into two categories.  
    _Example_: Spam Email Detector.
  - **Multi-Class Classifier**: Classifies data into more than two categories.

---

### Unsupervised Learning Models

#### Clustering:
- Groups objects with similar characteristics into one cluster, and objects with dissimilar characteristics into another.

#### Association:
- Identifies relationships between variables in large datasets, determining which items tend to occur together.

---

### K-Means Clustering
- **Working**: The algorithm assigns each data point to a cluster based on similarity. We specify the number of clusters, and the algorithm repeats this assignment process, finding patterns.

---

### Data Visualization with Matplotlib

- **Usage**:  
  Plots data (e.g., sepal length and width) according to labels assigned by the algorithm. Each label (type) is represented by a color using the `viridis` function.

- **Centers of Clusters**:  
  The center (mean) of each cluster is marked on the plot with an 'X', indicating the average values of the data points in that cluster.

---

### Deep Learning & Neural Networks

- **Overview**:  
  Neural networks aim to mimic human thinking to solve complex problems.  
  _Example_: Driverless cars require large labeled datasets for accurate performance.

- **Process**:
  1. **Architecture Definition**: Specify the number of layers and neurons.
  2. **Forward Propagation**: Data is passed through layers to make predictions.
  3. **Loss Function**: Measures the accuracy of predictions compared to actual values.
  4. **Backpropagation**: Calculates gradients and adjusts weights to minimize the loss.
  5. **Optimization**: Uses gradients to continuously improve the model’s performance.

---

### Convolutional Neural Networks (CNN)

- **Purpose**:  
  CNNs are deep learning models specialized for processing and analyzing visual data (e.g., images). They detect patterns such as edges, textures, and shapes.

- **Key Components**:
  1. **Convolutional Layer**:  
     Extracts features from images using filters (kernels) that apply dot products to form feature maps.
  
  2. **Max Pooling Layer**:  
     Reduces the dimensionality of feature maps, making the model more efficient while retaining essential features.
  
  3. **Activation Function**:
     - **ReLU (Rectified Linear Unit)**: Enhances computational efficiency by converting negative values to 0, while leaving positive values unchanged.  
     - **Sigmoid**: Outputs values between 0 and 1 but suffers from the vanishing gradient problem.

---

### Sentiment Analysis Using NLP and RNN

- **Definition**:  
  Sentiment analysis uses natural language processing (NLP) techniques and Recurrent Neural Networks (RNN) to understand the sentiment (positive, negative, or neutral) of an input text.
