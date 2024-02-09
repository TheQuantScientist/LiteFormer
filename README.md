# Lightweight Transformer 
This repository contains the implementation of a novel lightweight Transformer model designed specifically for the task of univariate stock price forecasting. Leveraging the advanced capabilities of Transformer architectures, traditionally renowned for their success in natural language processing (NLP), our model adapts this powerful mechanism to the time series forecasting domain, focusing on predicting the future closing prices of stocks (e.g. IBM, AMZN, INTC, CSCO) with high accuracy and efficiency at minimal computational power. 

## Project Overview

Witnessing the growth and application of AI in live trading machines of the financial industry, this research proposes a lightweight Transformer model with meticulous architecture consisting mainly of positional encoding and renowned training techniques to mitigate model overfitting, hence offering prompt forecasting results through a univariate approach on the closing price of stocks. Employing MSE for loss alongside MAE and RMSE as core evaluation metrics, the proposed Transformer consistently surpasses renowned time series analysis models such as LSTM, LSTM-RNN, and SVR, averaging a reduction in forecasting errors by over 20\%. Likewise, the single-step Transformer is justified to be the most efficient model among others. After being trained across AMZN, INTC, CSCO, and IBM 20-year daily stock datasets, the Transformer demonstrates a high degree of accuracy in capturing instant downturn shocks, cyclical or seasonal patterns, and long-term dependencies. Thus, it only takes the model 19.36 seconds to generate forecasting results on a non-high-end local machine, fitting into the 1-minute trading window.

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

