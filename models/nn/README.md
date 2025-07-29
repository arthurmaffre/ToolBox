🚀 Neural Network Module (PyTorch)

This sub-directory contains a professionally structured, modular Neural Network class implemented in PyTorch, along with an easy-to-use testing script to quickly verify and evaluate model performance.

📂 File Structure
	•	nn.py: Contains the modular, highly customizable PyTorch neural network class.
	•	nn_test.py: Provides an example of how to test and validate the neural network on the MNIST dataset.

⚙️ Setup

Install Dependencies

Ensure you have PyTorch and torchvision installed:

pip install torch torchvision

🚧 Usage

Neural Network (nn.py)

The provided ModularNN class allows you to quickly create a flexible neural network model with custom options:

from nn import ModularNN
import torch.nn.functional as F

# Initialize the model
model = ModularNN(
    input_dim=784,            # Input dimensions (28x28 pixels for MNIST)
    output_dim=10,            # Number of output classes
    hidden_dims=[256, 128],   # Custom hidden layer dimensions
    activation=F.relu,        # Activation function
    dropout=0.3               # Dropout rate
)

The model expects input tensors of shape (batch_size, input_dim) and produces outputs of shape (batch_size, output_dim). Typically, the output is raw logits suitable for classification using a softmax or cross-entropy loss.

Model Testing (nn_test.py)

Execute the testing script to train and evaluate your model easily:

python nn_test.py

This script:
	•	Downloads the MNIST dataset automatically.
	•	Trains the network over multiple epochs.
	•	Reports the accuracy and loss on the test set after each epoch.

📊 Output Distribution

The model returns raw logits by default. You can convert logits to a probability distribution using softmax:

import torch.nn.functional as F

logits = model(input_tensor)
probabilities = F.softmax(logits, dim=1)

🛠 Integration

Easily integrate this modular neural network into larger pipelines:
	•	Pass input tensors directly to the forward() method.
	•	Adjust the dimensions (input_dim, output_dim, and hidden_dims) as needed for your specific tasks.

⸻

Enjoy your modular PyTorch neural network implementation! 🚀