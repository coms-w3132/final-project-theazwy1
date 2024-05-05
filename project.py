import yfinance as yf
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

DATA_PATH = "msft_data.json"

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


def visualize_hist(msft_hist):
    """Visualize historical stock prices."""
    msft_hist.plot.line(y="Close", use_index=True)
    plt.show()


def prepare_data(msft_hist, predictors):
    """Prepare the dataset with features and targets."""
    data = msft_hist[["Close"]]
    data = data.rename(columns={'Close': 'Actual_Close'})
    data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    prev_hist = msft_hist.copy().shift(1) # Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
    data = data.join(prev_hist[predictors]).iloc[1:]
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


# Main execution
def main():
    msft_hist = download_msft_data(DATA_PATH)
    visualize_hist(msft_hist)
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    data = prepare_data(msft_hist, predictors)

    # Create train and test sets
    train_predictors, train_target, test_predictors, test_target = create_train_test_sets(data, predictors, "Target")

    # Create and train the RandomForest model
    model = create_random_forest(n_estimators=100, min_samples_split=200, random_state=1)
    train_model(model, train_predictors, train_target)

    # Evaluate and plot model precision
    evaluate_and_plot_precision(model, test_predictors, test_target)

if __name__ == "__main__":
    main()