"""A tiny CPU training loop with manually derived gradients."""
import numpy as np

def loss_and_grad(x, y, weight, bias):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not x.size:
        raise ValueError('non-empty matching vectors required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite([weight,bias]).all():
        raise ValueError('finite inputs required')
    residual = weight * x + bias - y
    return float(np.mean(residual ** 2)), float(2 * np.mean(x * residual)), float(2 * np.mean(residual))

def train(x, y, steps=200, rate=0.1):
    if type(steps) is not int or steps < 0 or not np.isfinite(rate) or rate <= 0:
        raise ValueError('non-negative steps and finite positive rate required')
    weight, bias, history = 0.0, 0.0, []
    for _ in range(steps):
        loss, grad_w, grad_b = loss_and_grad(x,y,weight,bias)
        history.append(loss)
        weight -= rate * grad_w
        bias -= rate * grad_b
    history.append(loss_and_grad(x,y,weight,bias)[0])
    return weight, bias, history

if __name__ == '__main__':
    x, y = np.array([0.,1.,2.]), np.array([1.,3.,5.])
    loss, grad_w, grad_b = loss_and_grad(x,y,0.,0.)
    weight, bias, history = train(x,y)
    print(f'initial loss={loss:.6f} dw={grad_w:.6f} db={grad_b:.6f}')
    print(f'after one step loss={history[1]:.6f}')
    print(f'final weight={weight:.6f} bias={bias:.6f} loss={history[-1]:.12f}')
