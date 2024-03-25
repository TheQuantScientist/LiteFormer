# Lightweight Transformer 
This repository contains the implementation of a novel lightweight Transformer model designed specifically for the task of univariate stock price forecasting. Leveraging the advanced capabilities of Transformer architectures, traditionally renowned for their success in natural language processing (NLP), our model adapts this powerful mechanism to the time series forecasting domain, focusing on predicting the future closing prices of stocks (e.g. IBM, AMZN, INTC, CSCO) with high accuracy and efficiency at minimal computational power. 

## Project Overview

Witnessing the growth and application of AI in live trading machines of the financial industry, this research proposes a lightweight Transformer model with meticulous architecture consisting mainly of positional encoding and renowned training techniques to mitigate model overfitting, hence offering prompt forecasting results through a univariate approach on the closing price of stocks. Employing MSE for loss alongside MAE and RMSE as core evaluation metrics, the proposed Transformer consistently surpasses renowned time series analysis models such as LSTM, LSTM-RNN, and SVR, averaging a reduction in forecasting errors by over 50\%. Likewise, the single-step Transformer is justified to be the most efficient model among others. After being trained across AMZN, INTC, CSCO, and IBM 20-year daily stock datasets, the Transformer demonstrates a high degree of accuracy in capturing instant downturn shocks, cyclical or seasonal patterns, and long-term dependencies. Thus, it only takes the model 19.36 seconds to generate forecasting results on a non-high-end local machine, fitting into the 1-minute trading window.

## Features

- **Lightweight and Optimized Architecture:** Specifically designed for stock price forecasting, this model reduces computational requirements without sacrificing accuracy, enabling its use on machines with limited processing power.
- **Univariate Time Series Forecasting:** Employs a focused approach on predicting the closing prices of stocks, utilizing historical data to forecast future prices with high precision.
- **Advanced Model Training Techniques:** Incorporates dropout regularization, layer normalization, and early stopping to fine-tune the training process, enhancing model performance and preventing overfitting.
- **Dynamic Learning Rate Adjustment:** Utilizes the OneCycleLR scheduler for optimal learning rate adjustments during training, facilitating faster convergence and improved model accuracy.
- **Positional Encoding:** Integrates temporal information into the model, allowing it to capture time-dependent patterns in the stock market data effectively.
- **Reproducibility and Consistency:** Ensures reliable and reproducible results through fixed random seed initialization and detailed documentation of the data processing and model training pipeline.

## Technologies Used

- **Python:** The primary programming language used for implementing the model and preprocessing data.
- **PyTorch:** A powerful, flexible deep learning library utilized for building and training the Transformer model.
- **Pandas:** For efficient data manipulation and analysis, particularly useful for handling time series data.
- **Scikit-learn:** Employed for data preprocessing tasks, such as scaling and normalization, to prepare data for model training.
- **NumPy:** Essential for handling numerical operations, array manipulations, and transformations.
- **Matplotlib/Seaborn (Optional):** For visualizing forecasting results and model performance, enhancing interpretability and analysis.
- **Torch.optim:** Provides the AdamW optimizer, a variant of the Adam optimizer with improved weight regularization for training.
- **Torch.utils.data DataLoader:** Facilitates efficient batching, shuffling, and loading of data during the model training and evaluation process.

## Getting Started

Follow the instructions in the subsequent sections to set up your environment, train the model with your dataset, and evaluate its performance on stock price forecasting tasks through evaluation metrics of MAE and RMSE. This model has been tested on various datasets of prominent technology companies, demonstrating its capability to accurately capture market trends and instant downtrends, surpassing RNNs-based model.

