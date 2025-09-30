# Neural Network Library

A comprehensive neural network library implemented from scratch in Python, featuring both traditional machine learning and deep learning capabilities, along with reinforcement learning components.

## Project Overview

This project represents a complete neural network implementation built from the ground up, demonstrating deep understanding of machine learning fundamentals. The library includes:

- **Custom Neural Network Framework**: Complete implementation of neural networks with various layer types
- **Deep Learning Components**: Convolutional layers, pooling layers, and image processing capabilities
- **Reinforcement Learning**: Deep Q-Network (DQN) implementation for agent-based learning
- **Multiple Optimizers**: Adam, Gradient Descent, Momentum, and Mini-batch Gradient Descent
- **Comprehensive Metrics**: Accuracy, loss functions, and evaluation metrics
- **Image Classification**: Cat vs Dog classification using CNN architecture
- **Medical Diagnosis**: Breast cancer prediction using traditional neural networks

## Project Structure

```
NN_Library/
├── Library/                          # Core neural network library
│   ├── Layers/                       # Neural network layer implementations
│   │   ├── Layer_Dense.py           # Dense/fully connected layers
│   │   ├── Convolutional_Layer.py   # Convolutional neural network layers
│   │   ├── MaxPool2D.py             # Max pooling layers
│   │   ├── AvgPool2D.py             # Average pooling layers
│   │   └── Flatten.py               # Flatten layer for CNN to Dense transition
│   ├── Models/                      # Model architectures
│   │   ├── Model.py                 # Base model class
│   │   └── Sequential.py            # Sequential model implementation
│   ├── Optimizers/                  # Optimization algorithms
│   │   ├── Adam.py                  # Adam optimizer
│   │   ├── Gradient_Descent.py      # Standard gradient descent
│   │   ├── Momentum.py              # Momentum optimizer
│   │   └── Mini_batch_gradient_descent.py
│   └── metrics/                     # Evaluation metrics
│       └── Metrics.py              # Accuracy, loss functions
├── Image_Manipulation/              # Image processing utilities
│   ├── image_data_generator.py      # Custom data generator for images
│   ├── process_images.py            # Image preprocessing
│   └── populate_Image_Files.py      # Dataset organization
├── Reinforcement_Learning/          # RL components
│   ├── Agent/                       # Agent implementations
│   │   ├── Agent.py                 # Base agent class
│   │   └── DQN.py                   # Deep Q-Network implementation
│   ├── Environements/               # RL environments
│   │   └── CartPoleEnv.py          # CartPole environment wrapper
│   ├── ReplayMemory/               # Experience replay
│   │   └── ReplayMemory.py         # Memory buffer for DQN
│   └── Stratergies/               # Exploration strategies
│       └── epsilonGreedyStrategies.py
├── Utilities/                      # Helper utilities
│   ├── Timing/                     # Performance timing decorators
│   ├── Logging/                    # Logging utilities
│   └── Additional_functions/       # Utility functions
├── Data/                          # Datasets
│   ├── breastCancer.csv           # Breast cancer dataset
│   └── processed_images/          # Cat vs Dog image dataset
│       ├── train/                # Training images
│       ├── test/                 # Test images
│       └── valid/                # Validation images
├── breastCancer.py               # Breast cancer prediction example
├── CatsVsDogs.py                 # Cat vs Dog classification example
└── reinforcement_learning_cart_pole.py  # RL CartPole example
```

## Core Features

### Neural Network Layers
- **Dense Layers**: Fully connected layers with customizable activation functions
- **Convolutional Layers**: 2D convolution with padding and bias options
- **Pooling Layers**: Max and Average pooling for dimensionality reduction
- **Flatten Layer**: Converts multi-dimensional data to 1D for dense layers

### Optimization Algorithms
- **Adam Optimizer**: Adaptive learning rate with momentum
- **Gradient Descent**: Standard gradient descent with configurable learning rate
- **Momentum**: Gradient descent with momentum for faster convergence
- **Mini-batch**: Efficient batch processing for large datasets

### Activation Functions
- ReLU, Sigmoid, Tanh, Softmax
- Customizable activation functions for different layer types

### Loss Functions
- Categorical Cross-Entropy
- Mean Squared Error
- Binary Cross-Entropy

## 🎯 Example Applications

### 1. Breast Cancer Prediction (`breastCancer.py`)
```python
# Binary classification for medical diagnosis
model = Sequential([
    Layer_Dense(30, 20, optimizer=Adam(learning_rate)),
    Layer_Dense(20, 10, optimizer=Adam(learning_rate)),
    Layer_Dense(10, 2, activation='softmax', optimizer=Adam(learning_rate))
])
```
- **Dataset**: Wisconsin Breast Cancer Dataset
- **Architecture**: 3-layer dense network
- **Features**: 30 input features, binary classification
- **Performance**: Interactive training with validation split

### 2. Cat vs Dog Classification (`CatsVsDogs.py`)
```python
# CNN for image classification
model = Sequential([
    Convolutional_Layer(1, 8, (3,3), optimizer=Adam(learning_rate)),
    MaxPool2D((3,3), 2),
    Convolutional_Layer(8, 32, (3,3), optimizer=Adam(learning_rate)),
    MaxPool2D((3,3), 3),
    Flatten(),
    Layer_Dense(46208, 100, optimizer=Adam(learning_rate)),
    Layer_Dense(100, 10, optimizer=Adam(learning_rate)),
    Layer_Dense(10, 2, activation='softmax', optimizer=Adam(learning_rate))
])
```
- **Dataset**: Custom cat/dog image dataset (224x224 grayscale)
- **Architecture**: CNN with pooling and dense layers
- **Features**: Image preprocessing and data augmentation
- **Training**: Batch processing with validation

### 3. Reinforcement Learning (`reinforcement_learning_cart_pole.py`)
- **Environment**: OpenAI Gym CartPole
- **Algorithm**: Deep Q-Network (DQN)
- **Features**: Experience replay, epsilon-greedy exploration
- **Agent**: Autonomous learning agent

## 🛠️ Technical Implementation

### Key Design Patterns
- **Object-Oriented Design**: Modular layer-based architecture
- **Decorator Pattern**: Timing and logging decorators for performance monitoring
- **Strategy Pattern**: Pluggable optimizers and activation functions
- **Template Method**: Base classes with customizable implementations

### Performance Features
- **Vectorized Operations**: NumPy-based computations for efficiency
- **Memory Management**: Efficient weight and bias storage
- **Batch Processing**: Mini-batch gradient descent support
- **Timing Decorators**: Performance monitoring utilities

### Data Processing
- **Image Preprocessing**: Resize, grayscale conversion, normalization
- **Data Augmentation**: Custom image data generator
- **Dataset Organization**: Train/validation/test splits
- **CSV Processing**: Pandas-based data handling

## 📊 Dataset Information

### Breast Cancer Dataset
- **Source**: Wisconsin Breast Cancer Dataset
- **Features**: 30 numerical features
- **Classes**: 2 (Malignant/Benign)
- **Samples**: ~569 instances
- **Preprocessing**: Standardization (mean=0, std=1)

### Cat vs Dog Dataset
- **Images**: Custom processed dataset
- **Resolution**: 224x224 pixels
- **Format**: Grayscale
- **Split**: Train/Validation/Test
- **Classes**: Cat, Dog
- **Total Images**: ~300+ images

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas pillow
```

### Running Examples

1. **Breast Cancer Prediction**:
```bash
python breastCancer.py
```

2. **Cat vs Dog Classification**:
```bash
python CatsVsDogs.py
```

3. **Reinforcement Learning**:
```bash
python reinforcement_learning_cart_pole.py
```

### Custom Model Creation
```python
from Library.Models.Sequential import Sequential
from Library.Layers.Layer_Dense import Layer_Dense
from Library.Optimizers.Adam import Adam

# Create a simple neural network
model = Sequential([
    Layer_Dense(784, 128, optimizer=Adam(0.001)),
    Layer_Dense(128, 64, optimizer=Adam(0.001)),
    Layer_Dense(64, 10, activation='softmax', optimizer=Adam(0.001))
])

# Compile and train
model.compile(loss_function=metrics.categorical_crossEntropy, 
              metrics=[metrics.accuracy])
model.fit(epochs=10, batch_size=32, X_train, y_train, validation_split=0.2)
```

## 🔧 Advanced Features

### Custom Layers
- Extensible layer architecture
- Custom activation functions
- Configurable weight initialization

### Optimization
- Multiple optimizer implementations
- Learning rate scheduling
- Gradient clipping support

### Monitoring
- Real-time training metrics
- Performance timing
- Comprehensive logging

## Future Enhancements

- [ ] Dropout regularization
- [ ] Batch normalization
- [ ] More activation functions
- [ ] Model saving/loading
- [ ] GPU acceleration
- [ ] Additional RL algorithms
- [ ] Web interface
- [ ] Database integration

## Contributing

This project demonstrates fundamental understanding of neural networks and machine learning. Contributions are welcome for:
- Additional layer types
- New optimization algorithms
- Performance improvements
- Documentation enhancements

## Learning Outcomes

This project showcases:
- Deep understanding of neural network mathematics
- Implementation of backpropagation algorithms
- Convolutional neural network architecture
- Reinforcement learning concepts
- Image processing and computer vision
- Software engineering best practices
- Performance optimization techniques

---

**Note**: This library is built for educational purposes to demonstrate understanding of neural network fundamentals. For production use, consider established frameworks like TensorFlow or PyTorch.