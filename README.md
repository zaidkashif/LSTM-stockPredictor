# LSTM Stock Predictor

A deep learning project that uses Long Short-Term Memory (LSTM) neural networks to predict S&P 500 stock prices using historical data and technical indicators.

## 🚀 Project Overview

This project implements an LSTM-based neural network to predict stock prices by analyzing historical S&P 500 data along with technical indicators. The model uses a multi-feature approach, incorporating closing prices, moving averages, volume changes, and RSI (Relative Strength Index) to make more accurate predictions.

## 📊 Features

- **Multi-feature Prediction**: Uses 5 key features for enhanced prediction accuracy
- **Technical Indicators**: Incorporates SMA (10, 50), Volume Change, and RSI-14
- **Deep Learning**: PyTorch-based LSTM implementation with dropout for regularization
- **Data Visualization**: Comprehensive plotting of actual vs predicted prices
- **Model Persistence**: Save/load trained models for inference
- **Performance Metrics**: RMSE and MAE evaluation metrics

## 🏗️ Project Structure

```
LSTM-StockPredictor/
├── data/
│   ├── sp500.csv                    # Raw S&P 500 historical data
│   └── enriched_stock_data.csv      # Processed data with technical indicators
├── src/
│   ├── data_loader.py              # Data loading and preprocessing utilities
│   ├── model.py                    # LSTM neural network architecture
│   ├── train.py                    # Training utilities and sequence creation
│   └── tempCodeRunnerFile.py       # Temporary execution file
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Data analysis and visualization
│   ├── 02_modeling_lstm.ipynb      # Model development and experimentation
│   └── scaler.save                 # Saved MinMaxScaler for data normalization
├── outputs/
│   ├── plots/                      # Generated visualization plots
│   │   ├── actual_vs_predicted.png
│   │   ├── trainingloss.png
│   │   └── ...                     # Various prediction comparisons
│   └── saved_models/
│       └── lstm_stock_model.pth    # Trained PyTorch model
├── main.py                         # Main inference script
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore configuration
└── README.md                       # This file
```

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, CUDA support
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Source**: yfinance for stock data
- **Development**: Jupyter Notebooks
- **Environment**: Python 3.11+

## 📈 Model Architecture

The LSTM model consists of:
- **Input Layer**: 5 features (Close, SMA_10, SMA_50, Volume_change, RSI_14)
- **LSTM Layers**: 2-layer LSTM with 128 hidden units each
- **Dropout**: 20% dropout for regularization
- **Output Layer**: Single dense layer for price prediction
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam optimizer

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU (optional, for faster training)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LSTM-StockPredictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv lstmenv
   # On Windows
   lstmenv\Scripts\activate
   # On Unix/macOS
   source lstmenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data availability**
   - The `data/sp500.csv` file should contain S&P 500 historical data
   - Data format: Date, Close, High, Low, Open, Volume

## 🚀 Usage

### 1. Data Preprocessing & Training (Jupyter Notebooks)

Run the Jupyter notebooks in sequence:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_modeling_lstm.ipynb
```

### 2. Direct Inference (Main Script)

For quick inference with the pre-trained model:

```bash
python main.py
```

This will:
- Load the pre-trained model from `outputs/saved_models/lstm_stock_model.pth`
- Process the test data
- Generate predictions
- Create visualization plots
- Display RMSE and MAE metrics

### 3. Custom Training

```python
from src.data_loader import load_and_preprocess_data
from src.train import create_sequences, train_model
from src.model import StockLSTM

# Load and preprocess data
df = load_and_preprocess_data('data/sp500.csv')
features = df[['Close', 'SMA_10', 'SMA_50', 'Volume_change', 'RSI_14']]

# Create sequences and train model
X, y = create_sequences(scaled_data, sequence_length=5)
model = StockLSTM(input_size=5)
trained_model, losses = train_model(model, X_train, y_train)
```

## 📊 Data Features

The model uses the following engineered features:

| Feature | Description |
|---------|-------------|
| **Close** | Stock closing price (primary target) |
| **SMA_10** | 10-day Simple Moving Average |
| **SMA_50** | 50-day Simple Moving Average |
| **Volume_change** | Percentage change in trading volume |
| **RSI_14** | 14-day Relative Strength Index |

## 📈 Model Performance

The model's performance is evaluated using:
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **Visual Analysis**: Actual vs Predicted price plots

## 📁 Key Files Description

### Core Scripts
- **`main.py`**: Main inference script for testing the trained model
- **`src/model.py`**: LSTM neural network architecture definition
- **`src/train.py`**: Training utilities, sequence creation, and model saving
- **`src/data_loader.py`**: Data loading, preprocessing, and feature engineering

### Data Files
- **`data/sp500.csv`**: Historical S&P 500 data (2010-present)
- **`notebooks/scaler.save`**: Saved MinMaxScaler for consistent data normalization

### Output Files
- **`outputs/saved_models/lstm_stock_model.pth`**: Trained PyTorch model
- **`outputs/plots/`**: Generated visualization plots

## 🔍 Model Training Process

1. **Data Loading**: Load S&P 500 historical data
2. **Feature Engineering**: Calculate technical indicators (SMA, RSI, Volume change)
3. **Data Normalization**: Apply MinMaxScaler to features
4. **Sequence Creation**: Create time series sequences (window size: 5)
5. **Train/Test Split**: 80/20 split for training and evaluation
6. **Model Training**: Train LSTM with Adam optimizer
7. **Evaluation**: Generate predictions and calculate metrics
8. **Visualization**: Create actual vs predicted comparison plots

## ⚙️ Configuration Options

Key hyperparameters that can be modified:

```python
# Model Architecture
input_size = 5          # Number of input features
hidden_size = 128       # LSTM hidden units
num_layers = 2          # Number of LSTM layers
dropout = 0.2           # Dropout rate

# Training Parameters
sequence_length = 5     # Time series window size
batch_size = 32         # Training batch size
learning_rate = 1e-4    # Adam optimizer learning rate
num_epochs = 100        # Training epochs
```

## 🎯 Future Improvements

- [ ] Add more technical indicators (MACD, Bollinger Bands)
- [ ] Implement attention mechanism for better sequence modeling
- [ ] Add support for multiple stock symbols
- [ ] Implement real-time prediction pipeline
- [ ] Add hyperparameter tuning with Optuna
- [ ] Create web interface for predictions
- [ ] Add ensemble methods for improved accuracy

## 📊 Sample Results

The model generates various visualizations including:
- Training loss curves
- Actual vs Predicted price comparisons
- Last 200 days trend analysis
- Performance metrics summary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [LSTM Networks for Stock Prediction](https://www.researchgate.net/publication/research_on_lstm)
- [Technical Analysis Indicators](https://www.investopedia.com/technical-analysis/)

## 🙋‍♂️ Support

For questions or issues, please create an issue in the repository or contact the maintainer.

---

**Note**: This is a research/educational project. Stock market predictions are inherently uncertain, and this model should not be used for actual trading decisions without proper risk management and additional validation.
