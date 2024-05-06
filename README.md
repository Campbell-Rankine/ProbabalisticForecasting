# ProbabalisticForecasting
---

An implementation of the Lag-Llama model with a GluonTS dataloader integration. My notes about probabilistic decoder only transformers can be found in the Probabilistic_Transformers_Notes PDF.

### Downloading the data

By default the data download script will run using the current date and will attempt to download 650 different tickers over a timespan of 2 years. Some basic data processing/filtering happens in this step including retrieving and building the start date, providing an integer categorical variable representing a ticker name, and retrieving the mean and standard deviation of the entire timeseries. Additionally stocks whose closing price has a standard deviation of less than 0.5 are filtered out as they were generally missing data or not worth predicting. To run the download script:

```sh
    python download_data.py
```

### Running the training script

To run the script with default parameters use:
```sh
    python main.py
```

To see a full list of command line arguments use:
```sh
    python main.py --help
```

### Tensorboard Logging

If you wish to monitor the training run, in a different terminal cd to the ProbabilisticForecasting directory. Next run the following command:

```sh
    tensorboard --logdir="./logging"
```

Tensorboard logging is supported for the following:

- Gradient Norm: Scalar value representing the size of the loss functions gradient. Useful to monitor vanishing/exploding gradients.
- Epoch Loss: For all epochs in the training run, graph the loss at each training step
- Mean Epoch loss: Useful to watch the tendencies of the loss over the epoch.