from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def load_data():
    """Load the diabetes dataset and return the features and target values."""
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target


def split_data(X, y):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)
