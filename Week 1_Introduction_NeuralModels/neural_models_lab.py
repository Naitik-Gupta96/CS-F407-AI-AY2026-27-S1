"""
Neural Models Lab
Learning, Depth, Activations, and Output Layers

This file runs the XOR binary experiment, the zero-initialisation symmetry
experiment, the activation comparison, and the three-class extension.
"""

import numpy as np
import torch


# ----------------------------
# Data
# ----------------------------
X = torch.tensor(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]],
    dtype=torch.float32,
)

Y_XOR = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32).view(-1, 1)

# 0 = both inactive, 1 = disagreement, 2 = both active
Y_3CLASS = torch.tensor([0, 1, 1, 2], dtype=torch.long)


# ----------------------------
# Binary XOR model
# ----------------------------
class XORNet(torch.nn.Module):
    def __init__(self, activation="tanh"):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 1)
        self.activation = activation

    def hidden(self, z):
        if self.activation == "sigmoid":
            return torch.sigmoid(z)
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "relu":
            return torch.relu(z)
        raise ValueError("Unknown activation")

    def forward(self, x):
        return self.fc2(self.hidden(self.fc1(x)))


def train_binary(activation="tanh", seed=2, steps=3000, lr=0.05):
    torch.manual_seed(seed)

    model = XORNet(activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        initial_loss = float(loss_fn(model(X), Y_XOR))

    first_layer_gradient = None
    early_gradient_norm = None

    for step in range(steps):
        optimizer.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, Y_XOR)
        loss.backward()

        # Gradient after the first backward() call.
        if step == 0:
            first_layer_gradient = model.fc1.weight.grad.detach().clone()
            early_gradient_norm = torch.linalg.vector_norm(
                first_layer_gradient
            ).item()

        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits).view(-1)
        predictions = (probabilities >= 0.5).long()
        final_loss = float(loss_fn(logits, Y_XOR))

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "probabilities": probabilities,
        "predictions": predictions,
        "gradient": first_layer_gradient,
        "gradient_norm": early_gradient_norm,
    }


# ----------------------------
# Zero-initialisation symmetry
# ----------------------------
def zero_weight_symmetry_test(steps=5):
    torch.manual_seed(2)

    model = XORNet("tanh")

    # Set all weights and biases to zero.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for step in range(steps + 1):
        print(f"\nSymmetry step {step}")
        print(model.fc1.weight.detach())

        if step == steps:
            break

        optimizer.zero_grad()
        loss = loss_fn(model(X), Y_XOR)
        loss.backward()
        optimizer.step()


# ----------------------------
# Activation experiment
# ----------------------------
def activation_experiment():
    print("\nActivation experiment")

    for activation in ["sigmoid", "tanh", "relu"]:
        result = train_binary(activation=activation, seed=2)

        print(f"\n{activation}")
        print("Initial loss:", result["initial_loss"])
        print("Final loss:", result["final_loss"])
        print("Probabilities:", result["probabilities"].numpy())
        print("Predictions:", result["predictions"].numpy())
        print("Early ||grad W1||2:", result["gradient_norm"])


# ----------------------------
# Three-class extension
# ----------------------------
class ThreeClassNet(torch.nn.Module):
    def __init__(self, activation="tanh"):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 3)
        self.activation = activation

    def hidden(self, z):
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "sigmoid":
            return torch.sigmoid(z)
        if self.activation == "relu":
            return torch.relu(z)
        raise ValueError("Unknown activation")

    def forward(self, x):
        return self.fc2(self.hidden(self.fc1(x)))


def train_three_class(seed=2, steps=3000, lr=0.05):
    torch.manual_seed(seed)

    model = ThreeClassNet("tanh")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = float(loss_fn(model(X), Y_3CLASS))

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y_3CLASS)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        final_loss = float(loss_fn(logits, Y_3CLASS))

    print("\nThree-class extension")
    print("Final loss:", final_loss)
    print("Predicted classes:", predictions.numpy())
    print("Class probabilities:")
    print(probabilities.numpy())

    print("\nProbability sum for first example:",
          float(probabilities[0].sum()))

    # Optional numerical-stability diagnostic.
    shifted = torch.softmax(logits[0] + 100.0, dim=0)
    print("Maximum difference after adding 100 to all logits:",
          float(torch.max(torch.abs(shifted - probabilities[0]))))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("=== Binary XOR experiment ===")
    result = train_binary("tanh", seed=2)

    print("Initial loss:", result["initial_loss"])
    print("Final loss:", result["final_loss"])
    print("Probabilities:", result["probabilities"].numpy())
    print("Predictions:", result["predictions"].numpy())
    print("First-layer gradient after backward():")
    print(result["gradient"].numpy())
    print("Early ||grad W1||2:", result["gradient_norm"])

    activation_experiment()

    print("\n=== Zero-weight symmetry experiment ===")
    zero_weight_symmetry_test()

    train_three_class()
