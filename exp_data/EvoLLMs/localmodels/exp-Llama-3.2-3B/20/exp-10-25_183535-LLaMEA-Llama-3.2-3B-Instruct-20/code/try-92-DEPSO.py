import numpy as np
import random

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
            
            # Apply mutation with 20% probability
            if random.random() < self.mutation_prob:
                # Select two random lines
                j = random.randint(0, self.budget - 1)
                k = random.randint(0, self.budget - 1)
                
                # Calculate the mutation factor
                mu = np.random.uniform(0.5, 1.5, (self.dim,))
                
                # Apply mutation
                self.x[i] = self.x[i] + mu * (self.x[j] - self.x[i])
                
                # Check if the new position is better
                f = func(self.x[i])
                if f < self.f_best:
                    self.f_best = f
                    self.x_best = self.x[i]

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DEPSO(budget, dim)
for x in optimizer():
    func(x)