import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Data.data_preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LSTMNetwork(nn.Module):
    def __init__(self, features, num_classes, dropout_rate=0.2):
        super(LSTMNetwork, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=features,
            hidden_size=128,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(64, 32)  # Fully Connected Layer
        self.fc2 = nn.Linear(32, num_classes)  # Output Layer

    def forward(self, x):
        x, _ = self.lstm1(x)  # First LSTM layer (returns sequences)
        x, _ = self.lstm2(x)  # Second LSTM layer (returns last hidden state)
        # x = x[:, -1, :]  # Take the last timestep's output
        x = F.relu(self.fc1(x))  # Apply ReLU
        x = self.fc2(x)  # Output layer (raw logits, no softmax)
        return x


def main():

    X, y = DataPreprocessor.prepare_data_with_engineering(
        "R1", "all", "A", FeatureEngineering.engineer_features
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LSTMNetwork(features=21, num_classes=5)

    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train.values, dtype=torch.long)
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test.values, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    with torch.no_grad():
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_labels)
        logging.info(f"Test Loss: {test_loss.item():.4f}")


if __name__ == "__main__":
    main()
