"""Shapes, views, broadcasting, and matrix multiplication on CPU."""
import numpy as np

def demo():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    w = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
    bias = np.array([10, 20], dtype=np.float64)
    y = x @ w + bias
    return x, w, y

if __name__ == "__main__":
    x, w, y = demo()
    print("x.shape=", x.shape, "w.shape=", w.shape)
    print(y)
    print("x.strides=", x.strides, "transpose.strides=", x.T.strides)
    print("transpose shares memory:", np.shares_memory(x, x.T))
