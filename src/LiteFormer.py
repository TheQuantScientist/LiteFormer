import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import os
import uuid
import time
import psutil

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define stock file paths
stock_files = [
    
]

# Data Preprocessing
def load_and_preprocess_data(file_path, sequence_length, steps):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    data_close = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(data_close).flatten()
    return normalized_data, scaler, data_close

def create_sequences(data, sequence_length, steps):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - steps + 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length:i + sequence_length + steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Lightweight Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
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
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.decoder = nn.Linear(d_model, n_features)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = x.unsqueeze(-1).transpose(0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
            self.src_mask = mask
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output)
        return output[-1:]

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_transformer(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs, patience, device, stock_name):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    history = {'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': []}
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        train_mae, train_rmse = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            # Calculate MAE and RMSE for training
            predictions_np = predictions.view(-1).cpu().detach().numpy()
            labels_np = labels.view(-1).cpu().detach().numpy()
            train_mae.append(mean_absolute_error(labels_np, predictions_np))
            train_rmse.append(np.sqrt(mean_squared_error(labels_np, predictions_np)))
        
        # Average training metrics for the epoch
        history['train_mae'].append(np.mean(train_mae))
        history['train_rmse'].append(np.mean(train_rmse))
        
        # Evaluate on validation set
        val_loss, val_mae, val_rmse = evaluate_transformer(model, test_loader, criterion, device)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        
        print(f'Epoch {epoch+1}: Training Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss}, '
              f'Train MAE: {history["train_mae"][-1]:.6f}, Val MAE: {val_mae:.6f}, '
              f'Train RMSE: {history["train_rmse"][-1]:.6f}, Val RMSE: {val_rmse:.6f}')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    return history

def evaluate_transformer(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    mae_scores, rmse_scores = [], []
    with torch.no_grad():
        for batch in val_loader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            # Calculate MAE and RMSE
            predictions_np = predictions.view(-1).cpu().numpy()
            labels_np = labels.view(-1).numpy()
            mae_scores.append(mean_absolute_error(labels_np, predictions_np))
            rmse_scores.append(np.sqrt(mean_squared_error(labels_np, predictions_np)))
    
    return total_loss / len(val_loader), np.mean(mae_scores), np.mean(rmse_scores)

def run_transformer(data, scaler, original_data, stock_name, sequence_length=14, steps=1, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = int(len(data) * 0.9)
    train_data, test_data = data[:split], data[split:]
    X_train, y_train = create_sequences(train_data, sequence_length, steps)
    X_test, y_test = create_sequences(test_data, sequence_length, steps)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    model = TransformerModel(n_features=steps, d_model=128, n_heads=8, n_hidden=512, n_layers=4, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)
    history = train_transformer(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs, patience=5, device=device, stock_name=stock_name)
    
    # Measure inference time, CPU usage, and power usage
    model.eval()
    inference_times = []
    cpu_usages = []
    power_usages = []
    y_pred, y_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            predictions = model(sequences)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Measure CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            cpu_usages.append(cpu_usage)
            
            # Estimate power usage (approximation based on CPU usage)
            # Assuming a baseline power of 50W for idle and scaling with CPU usage
            power_usage = 50 + (cpu_usage / 100) * 150  # Max additional 150W at full CPU usage
            power_usages.append(power_usage)
            
            y_pred.extend(predictions.view(-1).cpu().numpy())
            y_true.extend(labels.view(-1).cpu().numpy())
    
    # Calculate average metrics
    avg_inference_time = np.mean(inference_times)
    avg_cpu_usage = np.mean(cpu_usages)
    avg_power_usage = np.mean(power_usages)
    
    test_loss, test_mae, test_rmse = evaluate_transformer(model, test_loader, criterion, device)
    y_pred = np.array(y_pred).reshape(-1, steps)
    y_true = np.array(y_true).reshape(-1, steps)
    
    return test_loss, test_mae, test_rmse, avg_inference_time, avg_cpu_usage, avg_power_usage

# Main Pipeline
def main():
    results = []
    
    for stock_file in stock_files:
        stock_name = stock_file.split('/')[-1].split('.')[0]
        print(f"\nProcessing stock: {stock_name}")
        data, scaler, original_data = load_and_preprocess_data(stock_file, sequence_length=14, steps=1)
        
        # Run Transformer
        print("Running Lightweight Transformer...")
        test_loss, mae, rmse, inference_time, cpu_usage, power_usage = run_transformer(data, scaler, original_data, stock_file, sequence_length=14, steps=1, epochs=50)
        results.append({
            'Stock': stock_name,
            'Model': 'Lightweight Transformer',
            'Test Loss': test_loss,
            'MAE': mae,
            'RMSE': rmse,
            'Inference Time (s)': inference_time,
            'CPU Usage (%)': cpu_usage,
            'Power Usage (W)': power_usage
        })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('liteformer_results_with_metrics.csv', index=False)
    print("\nResults saved to 'liteformer_results_with_metrics.csv'")

if __name__ == "__main__":
    main()