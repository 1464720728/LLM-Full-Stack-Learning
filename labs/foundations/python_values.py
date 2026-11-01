"""Observe aliases, shallow copies, and immutable values."""
from copy import deepcopy

def aliases():
    original = [[1, 2], [3, 4]]
    shallow = original.copy()
    deep = deepcopy(original)
    shallow[0].append(9)
    return original, shallow, deep

if __name__ == "__main__":
    a, b, c = aliases()
    print("original:", a)
    print("shallow:", b)
    print("deep:", c)
    print("inner shared:", a[0] is b[0])
