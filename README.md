# isThisGay-ML

A Python-based Multi-Layer Perceptron (MLP) implemented with NumPy to classify text based on the presence of words related to "gay" in various languages.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Customization](#customization)
- [Requirements](#requirements)
- [License](#license)

## Features

- **Multilingual Support**: Detects "gay" related words across multiple languages, including but not limited to English, Korean, German, Chinese, Japanese, Russian, and more.
- **Customizable Vocabulary**: Easily add or modify the list of words to enhance detection capabilities.
- **Three Hidden Layers**: Utilizes a deep MLP with three hidden layers for improved learning capacity.
- **Adam Optimizer**: Implements the Adam optimization algorithm for efficient and stable training.
- **Binary Classification**: Outputs a binary decision (`True`/`False`) based on the presence of related words.
- **Simple Integration**: Easily integrate into any Python project for text classification tasks.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Feralthedogg/IsThisGay-ML.git
    ```

2. **Navigate to the Project Directory**

    ```bash
    cd isThisGay-ML
    ```

3. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4. **Install Dependencies**

    This project relies solely on NumPy. If you don't have NumPy installed, you can install it using pip:

    ```bash
    pip install numpy
    ```

## Usage

1. **Prepare Your Data**

    Ensure that your training data (`train_texts`) and corresponding labels (`train_labels`) are properly defined. Each label should be `1` if the text contains any "gay" related word and `0` otherwise.

2. **Run the Script**

    Execute the Python script to train the MLP model and perform predictions:

    ```bash
    python isThisGay-ML.py
    ```

    The script will train the model and periodically print the loss. After training, it will evaluate the model on predefined test sentences and output the predictions.

## Configuration

### Hyperparameters

- **Input Dimension (`input_dim`)**: Number of features, determined by the length of the `vocab` list.
- **Hidden Layers**:
  - `hidden_dim1`: Number of neurons in the first hidden layer.
  - `hidden_dim2`: Number of neurons in the second hidden layer.
  - `hidden_dim3`: Number of neurons in the third hidden layer.
- **Output Dimension (`output_dim`)**: Number of output neurons. For binary classification, this is set to `1`.
- **Learning Rate (`learning_rate`)**: Set to `0.0001` for the Adam optimizer.
- **Epochs (`epochs`)**: Number of training iterations. Set to `45000` in the provided script.

### Adam Optimizer Parameters

- **Beta1 (`beta1`)**: Exponential decay rate for the first moment estimates. Set to `0.9`.
- **Beta2 (`beta2`)**: Exponential decay rate for the second moment estimates. Set to `0.999`.
- **Epsilon (`epsilon`)**: Small constant for numerical stability. Set to `1e-8`.

## Training

The training process involves the following steps:

1. **Forward Pass**: Compute the activations for each layer using ReLU activation for hidden layers and Sigmoid activation for the output layer.
2. **Loss Calculation**: Use Binary Cross-Entropy as the loss function.
3. **Backward Pass**: Compute gradients using backpropagation.
4. **Parameter Update**: Update weights and biases using the Adam optimizer.

During training, the script periodically prints the loss to monitor the training progress.

### Example Output

```
Epoch 5000/45000, Loss: 0.6931
Epoch 10000/45000, Loss: 0.6930
Epoch 15000/45000, Loss: 0.6930
...
Epoch 45000/45000, Loss: 0.6930
```

**Note**: Due to the simplicity of the dataset and the random initialization of weights, the loss may not decrease significantly. For meaningful training, a larger and more diverse dataset is recommended.

## Inference

After training, you can use the `isThisGay_ML` function to classify new texts.

### Example

```python
test_sentences = [
    "This event is gay!",
    "Just a normal day",
    "homoseksual context",
    "A sentence with gey term",
    "A completely unrelated sentence",
    "homofil support group",
    "No keywords here",
    "Gay a ni is happening tonight"
]

for s in test_sentences:
    pred = isThisGay_ML(s)
    print(f"Text: '{s}' -> Predicted Gay?: {pred}")
```

**Expected Output**:

```
Text: 'This event is gay!' -> Predicted Gay?: True
Text: 'Just a normal day' -> Predicted Gay?: False
Text: 'homoseksual context' -> Predicted Gay?: True
Text: 'A sentence with gey term' -> Predicted Gay?: True
Text: 'A completely unrelated sentence' -> Predicted Gay?: False
Text: 'homofil support group' -> Predicted Gay?: True
Text: 'No keywords here' -> Predicted Gay?: False
Text: 'Gay a ni is happening tonight' -> Predicted Gay?: True
```

**Note**: The predictions may not be accurate due to the limited dataset and lack of comprehensive training. For better performance, use a larger and more representative dataset.

## Customization

### Adding More Words

You can extend the `gayWords` list with more words or phrases to improve detection:

```python
gayWords = [
    "gay", "hii kɛ hii", "abagaala ebisiyaga", "gai", "gay rehegua",
    # ... existing words
    "newword1", "newword2",  # Add new words here
]
```

### Adjusting Hyperparameters

Modify the hyperparameters as needed to experiment with different model architectures and learning configurations:

```python
input_dim = len(vocab)
hidden_dim1 = 64  # Increase the number of neurons
hidden_dim2 = 32
hidden_dim3 = 16
learning_rate = 0.0001
epochs = 50000
```

### Implementing Advanced Features

For enhanced performance, consider implementing the following:

- **Batch Training**: Instead of processing the entire dataset at once, divide it into batches.
- **Regularization**: Add techniques like Dropout or L2 regularization to prevent overfitting.
- **Early Stopping**: Monitor the loss and stop training when it stops decreasing to save computational resources.
- **Data Augmentation**: Increase the diversity of the training data by augmenting it with variations.

## Requirements

- **Python**: Version 3.6 or higher
- **NumPy**: Version 1.18 or higher

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
