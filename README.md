# Boat Image Classification with CNNs

## Project Report

This project focused on classifying boat images using Convolutional Neural Networks (CNNs). We explored both a custom-built CNN and a pretrained model (MobileNet) to identify boat categories within a dataset of 4,774 images spanning 24 distinct classes.
## Data Set
https://drive.google.com/file/d/1dPu0jymKrDR34xtgLijOK_DgwxIRyOBk/view

## Data Preprocessing

The dataset, containing 4,774 boat images across 24 categories, was prepared for model training and evaluation through the following steps:

1.  **Data Access:** The dataset was accessed by mounting Google Drive in a Colab environment and extracting it from a shared link into a designated directory.
2.  **Data Splitting:** The dataset was divided into training and validation sets to facilitate model training and unbiased performance evaluation:
    * **Training set:** 3,820 images (80%)
    * **Validation set:** 954 images (20%)
3.  **Preprocessing:** Images underwent the following transformations:
    * **Resizing:** All images were resized to a fixed dimension to ensure compatibility with the CNN input layer.
    * **Normalization:** Pixel values were normalized to the range \[0, 1] to accelerate model convergence during training.

## Model Development & Performance

### Custom Sequential CNN Model

A sequential CNN model was developed with the following architecture:

* **Convolutional Layers:** Multiple convolutional layers with an increasing number of filters (32 to 128) were used for feature extraction from the images.
* **Max Pooling:** Max pooling layers were applied after convolutional layers to reduce the spatial dimensions of the feature maps and introduce translational invariance.
* **Dropout:** Dropout layers with varying dropout rates (0.25 to 0.5) were incorporated to mitigate overfitting by randomly setting a fraction of neuron outputs to zero during training.
* **Flattening Layer:** The output from the convolutional and pooling layers was flattened into a one-dimensional vector.
* **Dense Layers:** Fully connected (dense) layers were used for the final classification. The architecture included a dense layer with 128 ReLU (Rectified Linear Unit) activation neurons followed by an output dense layer with 24 neurons and a softmax activation function to produce probability distributions over the 24 boat categories.

The model was compiled using the Adam optimizer and the categorical crossentropy loss function, which is suitable for multi-class classification tasks. The model was trained for 5 epochs, and the performance metrics per epoch are summarized below:

| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
| :---- | :------- | :----- | :----------- | :------- |
| 1     | 13.71%   | 324.57 | 46.33%       | 2.4545   |
| 2     | 39.93%   | 2.2252 | 49.58%       | 2.4700   |
| 3     | 46.13%   | 1.9857 | 52.83%       | 2.1894   |
| 4     | 49.16%   | 1.8132 | 54.30%       | 2.1328   |
| 5     | 52.88%   | 1.6985 | 55.97%       | 2.0221   |

**Key Observations:**

* The custom CNN model demonstrated a steady improvement in both training accuracy (from 13.7% to 52.9%) and validation accuracy (from 46.3% to 56.0%) over the 5 epochs.
* The initial high loss value is likely due to the random initialization of the model's weights. The loss decreased and stabilized as the model learned from the training data.

### Improved Performance with Pretrained Model (MobileNet)

To explore the benefits of transfer learning, a pretrained MobileNet model was evaluated on the same dataset. The results after just 2 epochs were significantly better than the custom CNN:

| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
| :---- | :------- | :----- | :----------- | :------- |
| 1     | 44.55%   | 2.0085 | 67.30%       | 1.2083   |
| 2     | 61.14%   | 1.3544 | 66.67%       | 1.1089   |

**Advantages of Pretrained Model:**

* **Faster Training:** The pretrained model exhibited significantly faster training times per epoch (approximately 235 seconds) compared to the custom CNN (approximately 1,815 seconds). This is because the pretrained model starts with learned features, requiring less training to adapt to the new task.
* **Higher Validation Accuracy:** The MobileNet model achieved a higher validation accuracy (66.7%) compared to the custom CNN (56.0%), indicating better generalization and the ability to extract more relevant features.
* **Lower Loss Values:** The pretrained model also showed lower loss values on both the training and validation sets, suggesting improved model performance and stability.

### Test Prediction

The pretrained model was used to predict the category of a test image, and it successfully classified the image as "VaporettoACTV".

## Challenges Faced

Several challenges were encountered during the project:

* **Data Import:** Initially, there were difficulties in correctly mounting Google Drive and structuring the dataset directory within the Colab environment.
* **Training Time:** The training time for the custom CNN was extremely slow (around 30 minutes per epoch), likely due to limitations in available hardware resources and potentially unstable internet connectivity.
* **CNN Complexity:** Handling NumPy arrays for image data and understanding the interpretation of model predictions were initially challenging aspects of working with CNNs.

## Conclusion

The project demonstrated the fundamental workflow of image classification using CNNs, from data preprocessing to model evaluation. While the custom CNN model showed progressive learning, it was outperformed by the pretrained MobileNet model in terms of both training speed and classification accuracy.

**Key Takeaways:**

* Pretrained models can offer significant advantages in terms of training efficiency and performance, especially when dealing with limited datasets.
* Computational resources play a crucial role in the feasibility and speed of training deep learning models.
* Effective debugging of shape mismatches (e.g., `ValueError`) is essential for a smooth implementation of neural network models.

This project provided valuable foundational experience in CNN workflows and highlighted the practical considerations and challenges involved in deep learning for image classification tasks.
