import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.v = np.random.uniform(-1.0, 1.0, (budget, dim))
        self.f_best = np.inf
        self.x_best = None
        self.mutation_prob = 0.2

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the objective function
            f = func(self.x[i])
            
            # Update the personal best
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x[i]
                
            # Update the global best
            if f < func(self.x_best):
                self.f_best = f
                self.x_best = self.x[i]
                
            # Update the velocity
            self.v[i] = 0.5 * (self.v[i] + 0.5 * np.random.uniform(-1.0, 1.0, (self.dim,)))
            self.v[i] = self.v[i] + 1.0 * np.random.uniform(-1.0, 1.0, (self.dim,)) * (self.x[i] - self.x_best)
            self.v[i] = self.v[i] + 0.5 * np.random.uniform(-1.0, 1.0, (self.dim,)) * (self.x[i] - self.x[i])
            
            # Update the position
            self.x[i] = self.x[i] + self.v[i]
            
            # Apply mutation with probability 0.2
            if np.random.rand() < self.mutation_prob:
                self.x[i] = self.x[i] + np.random.uniform(-1.0, 1.0, (self.dim)) * np.random.uniform(-1.0, 1.0, (self.dim))

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DEPSO(budget, dim)
for x in optimizer(func):
    print(func(x))

# Test the BBOB test suite
import bbo

def test_bbo():
    from bbo import bbo

    # Define the objective function
    def func(x):
        return np.sum(x**2)

    # Initialize the optimizer
    optimizer = DEPSO(100, 10)

    # Test the objective function
    for i in range(24):
        x = np.random.uniform(-5.0, 5.0, 10)
        f = func(x)
        print(f"Function value at x = {x}: {f}")

# Run the test
test_bbo()