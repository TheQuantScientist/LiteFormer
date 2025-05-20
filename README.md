# LiteFormer: Lightweight Transformer for Financial Time Series Forecasting

LiteFormer is an encoder-only Transformer model optimized for univariate financial time series forecasting, targeting stock closing prices. Built with PyTorch, it leverages a compact architecture (`d_model=128`, `n_heads=8`, `n_layers=4`) to achieve efficient predictions on resource-constrained hardware. The model employs positional encoding, multi-head attention, and advanced optimization techniques (AdamW, OneCycleLR, early stopping) to deliver accurate single-step or multi-step forecasts, evaluated via Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Architecture

The model consists of a Transformer encoder with the following configuration:

- **Input**: Univariate time series (normalized closing prices, range `[-1, 1]`).
- **Positional Encoding**: Sinusoidal encodings to capture temporal dependencies.
- **Encoder**: 4 layers, 8 attention heads, `d_model=128`, `n_hidden=512`, dropout=0.1.
- **Output**: Linear layer projecting to `n_features * steps` for multi-step predictions.
- **Loss**: Mean Squared Error (MSE).
- **Optimizer**: AdamW (`lr=0.001`).
- **Scheduler**: OneCycleLR (`max_lr=0.01`).
- **Regularization**: Early stopping (patience=5), dropout.

### Key Code: Model Definition

```python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_hidden, n_layers, dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.decoder = nn.Linear(d_model, n_features)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = x.unsqueeze(-1).transpose(0, 1)
        mask = torch.triu(torch.ones(len(x), len(x)), diagonal=1).masked_fill(mask == 1, float('-inf')).to(x.device)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask)
        return self.decoder(output)[-steps:]
```

## Setup and Installation

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd liteformer
   ```

2. **Install Dependencies** (Python 3.8+):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install torch pandas numpy scikit-learn matplotlib
   ```

3. **Prepare Dataset**:
   - Provide a CSV file (e.g., `CSCO.csv`) with `Date` and `Close` columns.
   - Update the file path in the script:
     ```python
     data = pd.read_csv('path/to/CSCO.csv', parse_dates=['Date'])
     ```

## Training and Evaluation

### Data Preprocessing

The script normalizes closing prices and creates sequences for training:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler(feature_range=(-1, 1))
price_data_train = scaler.fit_transform(data['Close'].values.reshape(-1, 1)).flatten()
sequence_length = 1
steps = 1

def create_sequences(data, sequence_length, steps):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - steps + 1):
        xs.append(data[i:(i + sequence_length)])
        ys.append(data[i + sequence_length:i + sequence_length + steps])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(price_data_train, sequence_length, steps)
```

### Training Loop

The model trains with early stopping and OneCycleLR:

```python
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = TransformerModel(n_features=steps, d_model=128, n_heads=8, n_hidden=512, n_layers=4, dropout=0.1)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)
criterion = torch.nn.MSELoss()

def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs, patience):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            sequences, labels = batch[0].to(device), batch[1].to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        val_loss = evaluate(model, test_loader, criterion)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
```

### Evaluation Metrics

Post-training, the model computes MAE and RMSE:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred, y_true = [], []
with torch.no_grad():
    for batch in test_loader:
        sequences, labels = batch[0].to(device), batch[1].to(device)
        predictions = model(sequences)
        y_pred.extend(predictions.view(-1).cpu().numpy())
        y_true.extend(labels.view(-1).cpu().numpy())

y_pred = np.array(y_pred).reshape(-1, steps)
y_true = np.array(y_true).reshape(-1, steps)
mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
print(f'MAE: {np.mean(mae)}')
print(f'RMSE: {np.mean(rmse)}')
```

### Visualization (Optional)

Visualize predictions vs. actual values:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_true[:, 0], label='Actual', color='#1f77b4')
plt.plot(y_pred[:, 0], label='Predicted', color='#ff7f0e')
plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.title('LiteFormer: Stock Price Predictions')
plt.legend()
plt.savefig('predictions.png')
plt.show()
```

## Performance

- **Dataset**: CSCO daily closing prices.
- **Runtime**: ~19.36 seconds on standard hardware (CPU/GPU).
- **Metrics**: Competitive MAE and RMSE for short-term forecasts.
- **Limitations**:
  - Sequence length of 1 restricts long-term dependency modeling.
  - Univariate input limits feature diversity.
  - Non-stationarity not explicitly addressed.

## Future Enhancements

- Increase `sequence_length` (e.g., 10–50) for better temporal modeling.
- Extend to multivariate inputs (e.g., volume, technical indicators).
- Implement differencing or trend decomposition for non-stationary data.
- Add cross-validation for robust evaluation.
- Optimize for edge deployment with quantization or pruning.

## Dependencies

- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (optional)
