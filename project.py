import yfinance as yf
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

DATA_PATH = "msft_data.json"

# Download or load historical MSFT data
def download_msft_data(data_path):
    """Download or load historical MSFT data."""
    if os.path.exists(data_path):
        with open(data_path) as f:
            msft_hist = pd.read_json(data_path)
    else:
        msft = yf.Ticker("MSFT")
        msft_hist = msft.history(period="max")
        msft_hist.to_json(data_path)
    return msft_hist

# Visualize Microsoft stock prices
def visualize_hist(msft_hist):
    """Visualize historical MSFT stock prices."""
    msft_hist.plot.line(y="Close", use_index=True)
    plt.show()

# Prepare the dataset with features and targets
def prepare_data(msft_hist, predictors):
    """Prepare the dataset with features and targets."""
    data = msft_hist[["Close", "Open", "High", "Low", "Volume"]]
    data = data.rename(columns={'Close': 'Actual_Close'})
    data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    prev_hist = msft_hist.copy().shift(1)  # Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
    data = data.join(prev_hist[predictors], rsuffix="_prev").iloc[1:]
    return data

# Create a random forest classification model
def create_random_forest(n_estimators=100, min_samples_split=200, random_state=1):
    """Create a RandomForestClassifier with specified parameters."""
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)

# Create train and test datasets
def create_train_test_sets(data, predictors, target_column, test_size=100):
    """Create training and testing sets using the last `test_size` rows as the test set."""
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]
    return train[predictors], train[target_column], test[predictors], test[target_column]

# Train the model
def train_model(model, train_predictors, train_target):
    """Fit the RandomForest model to the training data."""
    model.fit(train_predictors, train_target)

# Evaluate model precision and print results
def evaluate_and_plot_precision(model, test_predictors, test_target):
    """Evaluate and plot the precision score of the model."""
    preds = model.predict(test_predictors)
    preds = pd.Series(preds, index=test_target.index)
    precision = precision_score(test_target, preds)
    print(f"Precision Score: {precision:.2f}")

    # Combine predictions and targets into a single DataFrame and plot
    combined = pd.concat({"Target": test_target, "Predictions": preds}, axis=1)
    combined.plot()
    plt.show()

    return precision

# Backtesting function
def backtest(data, model, predictors, target_column, start=1000, step=750, threshold=0.6):
    """Backtest the model using the specified parameters and a probability threshold."""
    predictions = []
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i + step].copy()

        train_predictors = train[predictors]
        train_target = train[target_column]
        test_predictors = test[predictors]
        test_target = test[target_column]

        # Fit the random forest model
        model.fit(train_predictors, train_target)

        # Make predictions with a probability threshold
        probs = model.predict_proba(test_predictors)[:, 1]  # Probabilities for class 1 (price up)
        preds = pd.Series((probs > threshold).astype(int), index=test.index)

        # Combine predictions and test values
        combined = pd.concat({"Target": test_target, "Predictions": preds}, axis=1)
        predictions.append(combined)

    return pd.concat(predictions)

# Evaluate model with backtesting
def evaluate_model(data, model, predictors, target_column, threshold=0.6):
    """Evaluate the model using backtesting."""
    predictions = backtest(data, model, predictors, target_column, start=1000, step=750, threshold=threshold)

    # Calculate precision
    precision = precision_score(predictions['Target'], predictions['Predictions'])
    print(f"Precision Score: {precision:.2f}")

    # Calculate value counts for predictions and actual target
    pred_value_counts = predictions['Predictions'].value_counts()
    target_value_counts = predictions['Target'].value_counts()
    print(f"Predictions Value Counts:\n{pred_value_counts}")
    print(f"Target Value Counts:\n{target_value_counts}")

    # Look at trades we would have made in the last 100 days
    predictions.iloc[-100:].plot()
    plt.show()

    return predictions

# Main execution
def main():
    msft_hist = download_msft_data(DATA_PATH)
    visualize_hist(msft_hist)
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    target_column = "Target"
    data = prepare_data(msft_hist, predictors)

    # Create train and test sets
    train_predictors, train_target, test_predictors, test_target = create_train_test_sets(data, predictors, target_column)

    # Create and train the RandomForest model
    model = create_random_forest(n_estimators=100, min_samples_split=200, random_state=1)
    train_model(model, train_predictors, train_target)

    # Evaluate and plot model precision
    evaluate_and_plot_precision(model, test_predictors, test_target)

    # Run the backtesting and evaluate model
    predictions = evaluate_model(data, model, predictors, target_column, threshold=0.6)

    # View first few predictions
    print(predictions.head())

if __name__ == "__main__":
    main()
