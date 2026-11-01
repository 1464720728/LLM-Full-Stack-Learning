"""Step 1: inspect every state update in a short loop."""
def total_with_trace(values):
    total, trace = 0, []
    for index, value in enumerate(values):
        before = total
        total = total + value
        trace.append((index, value, before, total))
    return total, trace

if __name__ == '__main__':
    total, trace = total_with_trace([3, 5, 0])
    print('index value before after')
    for row in trace:
        print(*row)
    print('total =', total)
