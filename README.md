# Forecasting Retail Sales with KANs

This project demonstrates forecasting retail sales using Kolmogorov-Arnold Networks (KANs).

## Business context

Kolmogorov-Arnold Networks (KANs) are neural network architectures based on the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be expressed as a sum of continuous univariate functions and an auxiliary function. This makes KANs highly expressive and well-suited for modeling complex, non-linear relationships in time series data.

KANs can represent complex, non-linear dependencies between variables. They help reduce dimensionality. They approximate any continuous function, making them flexible for various time series applications.

In this project, we apply KANs to a real-world time series forecasting task using U.S. retail sales data from FRED. We compare it head-to-head with ARIMA, the workhorse of classical forecasting, and the results are surprising.

## Article

Medium article: [Forecasting Retail Sales with Kolmogorov-Arnold Networks (KANs): Beating ARIMA with Deep Function Learning](https://medium.com/@kylejones_47003/forecasting-retail-sales-with-kolmogorov-arnold-networks-kans-beating-arima-with-deep-function-40c3f8d07fb2)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Date and sales columns
- Model parameters (lag, train_size)
- Output settings

## KANs (Kolmogorov-Arnold Networks)

KANs are a new type of neural network:
- Learn activation functions instead of weights
- More interpretable than traditional MLPs
- Can outperform ARIMA for time series

## Caveats

- By default, generates synthetic sales data.
- Full KAN implementation requires additional dependencies.
- Model performance depends on data quality and preprocessing.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).