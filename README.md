# Adaptive TSA Implementation

## Overview
This repository implements the Adaptive Time Series Analysis (TSA) method. It utilizes the IEEE standard 10-generator 39-node power system for data generation and applies machine learning models for analysis and visualization.

## Steps for Implementation

### 1. Data Generation
- Use the widely adopted IEEE standard 10-generator 39-node power system to generate the data.
- Normalize the generated data for further processing.

### 2. Run the Main Function
- Execute the `main_classification32.py` script to apply the Adaptive TSA method based on the shared feature extractor.

## Key Components

### `data_classification.py`
- **Description**: Handles data reading, normalization, and splitting the dataset into training and test sets.
  
### `backbones.py`
- **Description**: Defines the machine learning-based TSA model structure.

### `loss_funcs`
- **Description**: Implements the loss functions and updates the dynamic adversarial factor during training.

### `tSNE39HU.py`
- **Description**: Visualizes the results using dimensionality reduction (t-SNE).

## Requirements
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pytorch` (or `tensorflow` depending on the implementation)

## Usage
1. Clone this repository to your local machine.
3. Run the `main_classification32.py` script to analysis.

