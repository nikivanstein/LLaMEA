import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.v = np.random.uniform(-1.0, 1.0, (budget, dim))
        self.f_best = np.inf
        self.x_best = None
        self.refine_prob = 0.2

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
            
            # Refine the solution with probability 0.2
            if np.random.rand() < self.refine_prob:
                # Randomly select two individuals to refine
                idx1, idx2 = np.random.choice(self.budget, size=2, replace=False)
                # Calculate the average of the two individuals
                avg_x = (self.x[idx1] + self.x[idx2]) / 2
                # Calculate the average of the velocities
                avg_v = (self.v[idx1] + self.v[idx2]) / 2
                # Update the position and velocity
                self.x[i] = avg_x
                self.v[i] = avg_v

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DEPSO(budget, dim)
for x in optimizer():
    func(x)