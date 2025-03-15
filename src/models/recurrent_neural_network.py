import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
import time
import logging
from .base_model import BaseModel


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
        x = x[:, -1, :]  # Take the last timestep's output
        x = F.relu(self.fc1(x))  # Apply ReLU
        x = self.fc2(x)  # Output layer (raw logits, no softmax)
        return x


class LSTMModel(BaseModel):
    def __init__(self, input_shape, num_classes, lstm_units=50, dropout_rate=0.2):
        super().__init__("lstm")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

    def build_model(self):
        input_size = self.input_shape[1] if len(self.input_shape) > 1 else 1
        return LSTMNetwork(input_size, self.num_classes, self.dropout_rate).to(self.device)

    def train(self, X, y, test_size=0.3, random_state=42, epochs=50, batch_size=32):
        logging.info("Training LSTM model...")
        start_time = time.time()

        logging.debug(X)
        logging.debug(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logging.info(
            f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
        )

        inputs = torch.Tensor(X_train).to(self.device)
        labels = torch.Tensor(y_train.values).to(self.device)

        # Build model
        self.model = self.build_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        train_time = time.time() - start_time
        logging.info(f"LSTM training completed in {train_time:.2f} seconds")

        return X_train.cpu().numpy(), X_test, y_train.cpu().numpy(), y_test

    def evaluate(self, X_test, y_test, print_report=False):
        logging.info("\nEvaluating LSTM model...")

        # Convert to PyTorch tensors
        X_test_tensor = torch.tensor(X_test).to(self.device)

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        if print_report:
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))

        return accuracy, precision, recall, fscore
