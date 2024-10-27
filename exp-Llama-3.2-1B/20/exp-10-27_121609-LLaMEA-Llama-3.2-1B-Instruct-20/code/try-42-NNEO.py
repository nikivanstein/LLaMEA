# Description: Novel metaheuristic algorithm for black box optimization, using a combination of exploration-exploitation strategies and adaptative mutation.
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def objective_mutate(x):
            new_x = np.copy(x)
            for _ in range(2):  # adaptative mutation
                if np.random.rand() < 0.2:  # exploration
                    new_x[np.random.randint(self.dim)] += np.random.uniform(-1, 1)
                else:  # exploitation
                    new_x[np.random.randint(self.dim)] -= np.random.uniform(1, 2)
            return objective(new_x)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def evaluate_fitness(self, individual):
        new_individual = self.evaluate_fitness(individual)
        return new_individual

# Test the algorithm
func = lambda x: x**2  # black box function
nneo = NNEO(10, 10)  # 10 dimensions, 10 evaluations
nneo.population_size = 100  # increased population size for better exploration

# Initial solution
initial_solution = np.array([1.0, 1.0])
nneo.population = np.array([initial_solution])

# Run the algorithm
nneo(nneo, func)