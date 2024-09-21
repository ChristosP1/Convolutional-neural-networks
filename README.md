# Convolutional Neural Networks for Fashion-MNIST Classification

This project involves building a LeNet-based convolutional neural network (CNN) and exploring four variants to improve performance on the Fashion-MNIST dataset.

## Baseline Model

An adaptation of the original LeNet architecture with modern enhancements:

- **Activation Functions**: ReLU instead of Tanh.
- **Pooling Layers**: Max-pooling instead of average-pooling.
- **Architecture**:
![BaselineModel](./architectures/images/model_1.png)

  - **First Convolutional Layer**:
    - Input Channels: 1 (grayscale images)
    - Output Channels: 6
    - Kernel Size: 5x5
    - Stride: 1
    - Padding: 2 (to maintain 28x28 input size)
    - Followed by ReLU and max-pooling (reduces size to 14x14)
  - **Second Convolutional Layer**:
    - Input Channels: 6
    - Output Channels: 16
    - Kernel Size: 5x5
    - Stride: 1
    - Followed by ReLU and max-pooling (reduces size to 5x5)
  - **Fully Connected Layers**:
    - Layer 1: 120 units, ReLU activation
    - Layer 2: 84 units, ReLU activation
    - Output Layer: 10 units (for 10 classes), SoftMax activation
- **Weight Initialization**: Kaiming uniform initialization.

## Model Variants

### Variant 1: Adaptive Learning Rate

- **Change**: Learning rate halves every 5 epochs starting from 0.001.
- **Purpose**: Fine-tunes the model by reducing optimization steps over time.

### Variant 2: Increased Convolutional Filters

- **Change**: Increased filters from (6, 16) to (32, 64) in convolutional layers.
- **Purpose**: Allows the model to capture more complex features.

### Variant 3: Expanded Fully Connected Layers

- **Change**: Increased neurons in fully connected layers from (120, 84) to (200, 140).
- **Purpose**: Enhances the model's ability to process complex features.

### Variant 4: Batch Normalization

- **Change**: Added batch normalization after convolutional and fully connected layers.
- **Purpose**: Stabilizes learning and reduces sensitivity to initial conditions.

## Training and Validation

- **Dataset**: Fashion-MNIST
  - Training Set: 60,000 images
  - Test Set: 10,000 images
- **Preprocessing**:
  - Normalization using training set mean and standard deviation.
  - Data augmentation applied to training set.
- **Training Details**:
  - 15 epochs with a batch size of 32.
  - Early stopping based on highest validation accuracy.
  - Training set split: 80% training, 20% validation.

## Results


- **Best Model**: Variant 4 achieved the highest validation accuracy.
- **Test Set Evaluation**:
  - **Initial Best Model**: 92.5% accuracy.
  - **Retrained on Full Data**: 92.9% accuracy.
- **Observations**:
  - Consistent performance indicates good generalization.
  - Confusion matrices highlight improved classification, especially for similar classes like shirts and t-shirts.

## Choice Tasks

### 1. Adaptive Learning Rate Scheduler

Implemented `StepLR` scheduler in PyTorch:

- **Parameters**:
  - `step_size=5`: Updates every 5 epochs.
  - `gamma=0.5`: Learning rate is halved.
- **Effect**: Improved convergence and fine-tuning.

### 2. K-Fold Cross-Validation

- **Implementation**: 5-fold cross-validation on Variant 4.
- **Results**:
  - Average Training Accuracy: 95.4%
  - Average Validation Accuracy: 92.3%
- **Conclusion**: Model generalizes well across different data splits.

### 3. Auxiliary Output Layers

- **Addition**: Output layers after each convolutional block.
- **Purpose**: Provides insight into feature extraction at different network stages.
- **Findings**: Early layers focus on generic features; deeper layers specialize.

### 4. Data Augmentation Techniques

Applied the following to the training data:

- **RandomHorizontalFlip** (`p=0.25`): Simulates mirrored images.
- **ColorJitter** (`brightness=0.2`, `contrast=0.2`): Adjusts brightness and contrast.
- **RandomAutocontrast** (`p=0.2`): Enhances image contrast.
- **RandomSolarize** (`p=0.2`, `threshold=15`): Inverts pixel values above a threshold.

### 5. t-SNE Visualization

- **Visualization**: Applied t-SNE to the output of the penultimate layer.
- **Insights**:
  - Similar classes cluster together (e.g., shirts, coats).
  - Explains certain misclassifications in the model.

### Bonus: TensorBoard Visualization

- **Tool**: TensorBoard for live tracking.
- **Metrics Monitored**:
  - Training and validation loss.
  - Training and validation accuracy.
- **Benefit**: Real-time insights and comparisons between models.

## Model Weights and Resources

- **Models**: Baseline and all variants saved as `.pt` files.
- **Location**: [Model Weights Repository](https://github.com/ChristosP1/Convolutional-neural-networks/tree/main/models)
- **Additional Resources**:
  - Training graphs and logs.
  - Confusion matrices and t-SNE plots.
