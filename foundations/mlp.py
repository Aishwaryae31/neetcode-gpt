import numpy as np

from numpy.typing import NDArray
from typing import List


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        weights: List[NDArray[np.float64]],
        biases: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:

        for i in range(len(weights)):

            x = x @ weights[i] + biases[i]

            # ReLU only for hidden layers
            if i < len(weights) - 1:
                x = np.maximum(0, x)

        return np.round(x, 5)