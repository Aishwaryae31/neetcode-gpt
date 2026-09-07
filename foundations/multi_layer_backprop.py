import numpy as np

from typing import List


class Solution:

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert lists to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ---------- FORWARD ----------

        # Layer 1
        z1 = np.dot(W1, x) + b1

        # ReLU
        a1 = np.maximum(0, z1)

        # Layer 2
        predictions = np.dot(W2, a1) + b2

        # MSE Loss
        loss = np.mean((predictions - y_true) ** 2)

        # ---------- BACKWARD ----------

        # dLoss / dPrediction
        d_pred = 2 * (predictions - y_true) / len(y_true)

        # Gradients for W2 and b2
        dW2 = np.outer(d_pred, a1)
        db2 = d_pred

        # Gradient flowing back through W2
        da1 = np.dot(W2.T, d_pred)

        # ReLU derivative
        dz1 = da1 * (z1 > 0)

        # Gradients for W1 and b1
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }