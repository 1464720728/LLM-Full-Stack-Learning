"""Stable probabilities and a finite-difference derivative."""
import numpy as np

def softmax(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("non-empty finite vector required")
    weights = np.exp(x - x.max())
    return weights / weights.sum()

def finite_difference(function, x, h=1e-5):
    if h <= 0:
        raise ValueError("positive step required")
    return (function(x + h) - function(x - h)) / (2 * h)

if __name__ == "__main__":
    print("softmax:", softmax([1000, 1001, 1002]).round(6))
    print("derivative of x*x at 3:", round(finite_difference(lambda x: x*x, 3), 6))
