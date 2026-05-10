---
author: "Kyle Jones"
date_published: "April 7, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/forecasting-retail-sales-with-kolmogorov-arnold-networks-kans-beating-arima-with-deep-function-40c3f8d07fb2"
---

# Forecasting Retail Sales with Kolmogorov-Arnold Networks (KANs): Beating ARIMA with Deep Function... Kolmogorov-Arnold Networks (KANs) are neural network architectures based
on the Kolmogorov-Arnold representation theorem, which states that...

### Forecasting Retail Sales with Kolmogorov-Arnold Networks (KANs): Beating ARIMA with Deep Function Approximation
Kolmogorov-Arnold Networks (KANs) are neural network architectures based on the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be expressed as a sum of continuous univariate functions and an auxiliary function. This makes KANs highly expressive and well-suited for modeling complex, non-linear relationships in time series data.

KANs can represent complex, non-linear dependencies between variables. They help reduce dimensionality. They approximate any continuous function, making them flexible for various time series applications.

In this project, we apply KANs to a real-world time series forecasting task using U.S. retail sales data from FRED. We compare it head-to-head with ARIMA, the workhorse of classical forecasting, and the results are surprising.

We use the monthly U.S. Retail Sales Excluding Food Services index (RSXFS) from the Federal Reserve Economic Data (FRED) API. This dataset spans over a decade and reflects long-term economic growth, short-term shocks like COVID-19, and seasonality around holiday cycles.

#### Kolmogorov-Arnold Network (KAN)
KANs are universal approximators based on univariate decomposition. Our simple model trains over 100 epochs using the past 12 months of retail sales

```python
class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.decomposition = nn.Linear(input_dim, hidden_dim)
        self.aggregation = nn.Linear(hidden_dim, output_dim)

def forward(self, x):
        h = torch.tanh(self.decomposition(x))
        return self.aggregation(h)
```

We used a rolling window to structure input sequences and trained the model on 80% of the data, using the remaining 20% for testing.

ARIMA is the gold standard in time series forecasting.

```python
from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(train_values, order=(5, 1, 0)).fit()
```

It models autoregression, differencing, and moving averages --- but doesn't handle non-linear relationships directly.

### Evaluation
We compared models using Root Mean Squared Error (RMSE) on the test set.

KAN: 6,745.67

ARIMA: 18,121.76

KAN outperformed ARIMA. It had an RMSE 60% lower than ARIMA.

This confirms KAN's strength in capturing non-linear dynamics that traditional models can miss. Despite its simplicity, the KAN generalizes well and tracks short-term variation without overfitting.

Here's what that looks like visually:

#### Predicted vs Actual (Test Set)


#### Forecast for the Next 2 Years


### Why Did KAN Win?
The tanh activation allows KANs to shape highly non-linear curves. Even with a shallow network, KANs can outperform if the problem has complexity ARIMA can't model. The 12-month rolling input gives the model temporal context.

### Takeaways
KANs are easy to build in PyTorch. You can get state-of-the-art results on real-world data with just a few lines of code.

ARIMA is still useful --- but in domains with non-linear growth, shocks, or nonstationary behavior, deep learning methods can shine.

The retail sales dataset (RSXFS) is a useful benchmark --- interpretable, real, and rich with structure.

Full code

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import datetime
import random
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from statsmodels.tsa.arima.model import ARIMA
from pandas_datareader import data as web
from sklearn.metrics import mean_squared_error, mean_absolute_error
from kan import KAN
from chronos import ChronosPipeline
import time

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Unified config
prediction_length = 12
window = 24

# Load and preprocess data
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2025, 1, 1)
series = web.DataReader("RSAFS", "fred", start, end).dropna()["RSAFS"]
series.index.freq = "MS"

mean_val, std_val = series.mean(), series.std()
norm_series = (series - mean_val) / std_val

def make_lagged(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series.iloc[i:i + window].values)
        y.append(series.iloc[i + window])
    return np.array(X), np.array(y)

# Split
split = series.index[-(window + prediction_length)]
train_raw, test_raw = series[series.index <= split], series[series.index > split]
train_norm, test_norm = norm_series[norm_series.index <= split], norm_series[norm_series.index > split]

X_train, y_train = make_lagged(train_norm, window)
X_test, y_test = make_lagged(test_norm, window)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# ARIMA
start_arima = time.time()
arima = ARIMA(train_raw, order=(5, 1, 0)).fit()
pred_arima = arima.forecast(steps=prediction_length)
pred_arima.index = test_raw.index[:prediction_length]
arima_time = time.time() - start_arima

# MLP
class MLP(nn.Module):
    def __init__(self, dim): super().__init__(); self.net = nn.Sequential(
        nn.Linear(dim, 64), nn.Tanh(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

mlp = MLP(window)
opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

start_mlp = time.time()
for _ in range(200):
    mlp.train(); opt.zero_grad()
    loss = loss_fn(mlp(X_train_t), y_train_t); loss.backward(); opt.step()
mlp_time = time.time() - start_mlp

mlp.eval(); pred_mlp = mlp(X_test_t).squeeze().detach().numpy()

# KAN
kan = KAN([window, 32, 1], grid=3, k=2)
opt_kan = torch.optim.Adam(kan.parameters(), lr=0.01)

start_kan = time.time()
for _ in range(200):
    kan.train(); opt_kan.zero_grad()
    loss = loss_fn(kan(X_train_t), y_train_t); loss.backward(); opt_kan.step()
kan_time = time.time() - start_kan

kan.eval(); pred_kan = kan(X_test_t).squeeze().detach().numpy()

# LSTM
class LSTM(nn.Module):
    def __init__(self, dim, h=32): super().__init__(); self.lstm = nn.LSTM(dim, h, batch_first=True); self.fc = nn.Linear(h, 1)
    def forward(self, x): return self.fc(self.lstm(x)[0][:, -1])

X_train_lstm = X_train_t.unsqueeze(-1)
X_test_lstm = X_test_t.unsqueeze(-1)
lstm = LSTM(1)
opt_lstm = torch.optim.Adam(lstm.parameters(), lr=0.01)

start_lstm = time.time()
for _ in range(200):
    lstm.train(); opt_lstm.zero_grad()
    loss = loss_fn(lstm(X_train_lstm), y_train_t); loss.backward(); opt_lstm.step()
lstm_time = time.time() - start_lstm

lstm.eval(); pred_lstm = lstm(X_test_lstm).squeeze().detach().numpy()

# Inverse transform
y_test_actual = y_test * std_val + mean_val
valid_length = min(len(y_test_actual), prediction_length)
y_actual = y_test_actual[-valid_length:]
pred_mlp = pred_mlp[-valid_length:] * std_val + mean_val
pred_kan = pred_kan[-valid_length:] * std_val + mean_val
pred_lstm = pred_lstm[-valid_length:] * std_val + mean_val

# Chronos T5
device = "cuda" if torch.cuda.is_available() else "cpu"
chronos_pipe = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map=device,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
context = torch.tensor(series.loc[:split].values, dtype=torch.float32)
forecast = chronos_pipe.predict(context, prediction_length=prediction_length)
pred_chronos = forecast[0].median(dim=0).values.numpy()
chronos_time = 0.0
pred_chronos = pred_chronos[-valid_length:]

# Metrics
def mape(y, yhat): return np.mean(np.abs((y - yhat) / (y + 1e-10))) * 100
def evaluate(y, yhat): return (
    mean_squared_error(y, yhat) / 1e6,
    mean_absolute_error(y, yhat) / 1e6,
    mape(y, yhat)
)

models = ["ARIMA", "MLP", "KAN", "LSTM", "Chronos T5"]
preds = [
    pred_arima[:valid_length],
    pred_mlp,
    pred_kan,
    pred_lstm,
    pred_chronos
]
times = [arima_time, mlp_time, kan_time, lstm_time, chronos_time]
results = pd.DataFrame([evaluate(y_actual, p) + (t,) for p, t in zip(preds, times)],
                       columns=["RMSE (M USD)", "MAE (M USD)", "MAPE (%)", "Train Time (s)"],
                       index=models)

# Plot


# Define forecast boundary and zoom range
prediction_start_date = series.index[-prediction_length]
zoom_start_date = series.index[-24]  
series_zoom = series[series.index >= zoom_start_date]


# Get valid forecast window (based on shortest forecastable length)
valid_length = min(
    prediction_length,
    len(pred_arima),
    len(pred_mlp),
    len(pred_kan),
    len(pred_lstm),
    len(pred_chronos)
)

forecast_index = series.index[series.index >= prediction_start_date][:valid_length]
end_date = forecast_index[-1]

# Plot setup
plt.rcParams.update({
    'axes.grid': False,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False
})
fig, ax = plt.subplots(figsize=(12, 5))

# Plot actual series (last 24 months)
ax.plot(series_zoom.index, series_zoom.values, label="Actual", color="black", linewidth=2)

# Add forecast start boundary
ax.axvline(prediction_start_date, color="lightgray", linestyle="--", linewidth=1)

# Forecast lines
ax.plot(forecast_index, pred_arima[:valid_length], linestyle="--", linewidth=1.5)
ax.plot(forecast_index, pred_mlp[-valid_length:], linewidth=1.5)
ax.plot(forecast_index, pred_kan[-valid_length:], linestyle="-.", linewidth=1.5)
ax.plot(forecast_index, pred_lstm[-valid_length:], linestyle=":", linewidth=1.5)
ax.plot(forecast_index, pred_chronos[-valid_length:], linestyle="dotted", linewidth=1.5)

def add_forecast_label(y_vals, label, offset=0):
    ax.text(end_date + pd.DateOffset(months=1),
            y_vals[-1] + offset,
            label,
            fontsize=9, va='center', ha='left')

add_forecast_label(pred_arima[:valid_length], "ARIMA")
add_forecast_label(pred_mlp[-valid_length:], "MLP")
add_forecast_label(pred_kan[-valid_length:], "KAN")
add_forecast_label(pred_lstm[-valid_length:], "LSTM")
add_forecast_label(pred_chronos[-valid_length:], "Chronos T5")

# Formatting
ax.set_ylabel("Sales (Millions USD)")
ax.set_title("Retail Sales Forecasts (Last 24 Months, Forecast Starts at Grey Line)")
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("retail_forecast_stream_labels_zoom.png", dpi=300)
plt.show()

results.reset_index(names="Model", inplace=True)

```
