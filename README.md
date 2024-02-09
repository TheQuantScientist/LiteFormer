# Lightweight Transformer 
This repository contains the implementation of a novel lightweight Transformer model designed specifically for the task of univariate stock price forecasting. Leveraging the advanced capabilities of Transformer architectures, traditionally renowned for their success in natural language processing, our model adapts this powerful mechanism to the time series forecasting domain, focusing on predicting the future closing prices of stocks with high accuracy and efficiency.

## Project Overview

The project introduces a meticulously crafted Transformer model that integrates positional encoding and optimized architecture adjustments to address the challenges of time series analysis within the financial industry. Aimed at providing prompt and accurate forecasting results, this model emphasizes a univariate approach, focusing solely on the closing prices of stocks to predict their future values. The implementation is tailored to fit within the constraints of real-time trading applications, offering a solution that balances computational efficiency with predictive precision.

## Features

- **Lightweight Design:** Customized to ensure minimal computational overhead while maintaining high forecasting accuracy, making it suitable for deployment on non-high-end machines.
- **Univariate Forecasting:** Focuses on the closing price of stocks, employing a data-driven approach to capture and predict future price movements.
- **Efficient Training and Evaluation:** Utilizes advanced techniques such as dropout regularization, early stopping, and dynamic learning rate adjustments to optimize the training process and prevent overfitting.
- **Reproducibility:** Ensures consistent results across runs with fixed seeds for random number generators and a clear, step-by-step experimental setup.

## Technologies Used

- Python
- PyTorch for model implementation and training
- Pandas for data manipulation
- Scikit-learn for data preprocessing
- DataLoader from PyTorch for efficient batch processing

## Getting Started

Follow the instructions in the subsequent sections to set up your environment, train the model with your dataset, and evaluate its performance on stock price forecasting tasks. This model has been tested on various datasets, including prominent technology companies, demonstrating its capability to accurately capture market trends and fluctuations.

