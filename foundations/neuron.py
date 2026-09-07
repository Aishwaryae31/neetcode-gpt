import numpy as np
from numpy.typing import NDArray


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        activation: str
    ) -> float:

        # Step 1: weighted sum + bias
        z = np.dot(x, w) + b

        # Step 2: activation
        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))

        elif activation == "relu":
            output = np.maximum(0, z)

        return round(float(output), 5)