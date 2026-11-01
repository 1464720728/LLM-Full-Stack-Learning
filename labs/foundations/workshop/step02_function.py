"""Step 2: a function with an explicit input and output contract."""
import json

def summarize_tokens(values):
    count, total = 0, 0
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError('each token count must be a non-negative integer')
        count += 1
        total += value
    return {'count': count, 'tokens': total, 'mean_tokens': total / count if count else None}

if __name__ == '__main__':
    print(json.dumps(summarize_tokens([3, 5, 0]), indent=2))
