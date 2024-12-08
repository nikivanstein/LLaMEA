import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.best_solution = None

    def __call__(self, func):
        def objective(x):
            return func(x)
        
        # Initialize the population with random solutions
        for _ in range(100):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            if len(self.population) < self.budget:
                self.population.append((x, objective(x)))
            else:
                # Select the best solution from the current population
                idx = np.argsort([obj(x) for obj in self.population])[:self.budget]
                self.population = self.population[idx]
                # Refine the strategy by changing the probability of selecting the best solution
                if np.random.rand() < 0.45:
                    idx = np.random.choice(len(self.population), p=[0.5, 0.5])
                    self.population = [self.population[i] for i in idx]

        # Return the best solution found
        return self.population[0]

    def fitness(self, func, x):
        return func(x)

# Evaluate the BlackBoxOptimizer on the BBOB test suite
def evaluate_bbob():
    # Define the black box functions
    def f1(x):
        return np.sin(x) + np.cos(x)

    def f2(x):
        return np.sin(x) * np.cos(x)

    def f3(x):
        return x**2 + 2*x + 1

    def f4(x):
        return np.sin(x) * np.cos(x) + np.sin(x)

    # Define the objective functions
    def objective1(x):
        return f1(x)

    def objective2(x):
        return f2(x)

    def objective3(x):
        return f3(x)

    def objective4(x):
        return f4(x)

    # Create the BlackBoxOptimizer instances
    optimizer1 = BlackBoxOptimizer(budget=100, dim=2)
    optimizer2 = BlackBoxOptimizer(budget=100, dim=2)
    optimizer3 = BlackBoxOptimizer(budget=100, dim=2)
    optimizer4 = BlackBoxOptimizer(budget=100, dim=2)

    # Evaluate the objective functions
    print("Objective 1:", optimizer1.fitness(f1, 0))
    print("Objective 2:", optimizer2.fitness(f2, 0))
    print("Objective 3:", optimizer3.fitness(f3, 0))
    print("Objective 4:", optimizer4.fitness(f4, 0))

# Run the evaluation
evaluate_bbob()