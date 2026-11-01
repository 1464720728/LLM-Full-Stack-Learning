"""An object owns state and enforces its state-transition rules."""
from dataclasses import dataclass, field

@dataclass
class Budget:
    limit: int
    used: int = field(default=0, init=False)

    def __post_init__(self):
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("non-negative integer limit required")

    @property
    def remaining(self):
        return self.limit - self.used

    def consume(self, amount: int):
        if type(amount) is not int or amount < 0:
            raise ValueError("non-negative integer amount required")
        if amount > self.remaining:
            raise ValueError("budget exceeded")
        self.used += amount

if __name__ == "__main__":
    budget = Budget(10)
    budget.consume(3)
    budget.consume(4)
    print(budget, "remaining=", budget.remaining)
