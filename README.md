Adaptive TSA Implementation
To implement Adaptive TSA, follow these steps:
Data Generation: Use the widely used IEEE standard 10-generator 39-node power system to generate data, then normalize the data.
Run the Main Function: Execute the main function main_classification32.py to apply the Adaptive TSA method based on the shared feature extractor.

Key Components:
data_classification.py: Handles data reading, normalization, and the splitting of the dataset into training and test sets.
backbones.py: Defines the machine learning-based TSA model structure.
loss_funcs: Implements the loss functions and updates the dynamic adversarial factor.
tSNE39HU.py: Visualizes the results through dimensionality reduction.
