"""Reproduce a boundary bug before fixing the implementation."""
def broken_mean(values):
    return sum(values) / (len(values) - 1)

def mean(values):
    values = list(values)
    if not values:
        raise ValueError("empty sample")
    return sum(values) / len(values)

if __name__ == "__main__":
    data = [2, 4, 6]
    print("broken:", broken_mean(data))
    print("correct:", mean(data))
    print("single:", mean([9]))
