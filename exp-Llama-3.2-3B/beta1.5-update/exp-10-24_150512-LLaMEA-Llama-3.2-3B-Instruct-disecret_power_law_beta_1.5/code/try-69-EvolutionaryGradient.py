import numpy as np
import random
import time

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.f_values = np.zeros(self.population_size)
        self.gradient = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        start_time = time.time()
        for _ in range(self.budget):
            # Compute gradient of the objective function
            for i in range(self.population_size):
                gradient = np.zeros(self.dim)
                h = 1e-1
                for j in range(self.dim):
                    gradient[j] = (func(self.population[i] + h * np.eye(self.dim)[j]) - func(self.population[i] - h * np.eye(self.dim)[j])) / (2 * h)

                # Update the current solution using evolutionary strategy
                self.population[i] += 0.5 * np.random.normal(0, 0.1, size=self.dim)

                # Update the best solution
                f = func(self.population[i])
                if f < self.f_values[i]:
                    self.f_values[i] = f
                    self.population[i] = self.population[i].copy()
                    if f < self.f_best:
                        self.f_best = f
                        self.x_best = self.population[i].copy()

                # Add gradient information to the evolutionary strategy
                self.gradient[i] = gradient

                # Check for convergence
                if _ % 100 == 0 and np.all(np.abs(self.population - self.x_best) < 1e-6):
                    print("Converged after {} iterations".format(_))
                    break

            # Perform mutation
            for i in range(self.population_size):
                if random.random() < 0.025:
                    mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
                    self.population[i] += mutation

            # Perform crossover
            for i in range(self.population_size):
                if random.random() < 0.025:
                    j = random.randint(0, self.population_size - 1)
                    crossover = (self.population[i] + self.population[j]) / 2
                    self.population[i] = crossover

            # Check for termination
            if time.time() - start_time > 3600:
                break

# Example usage:
def func(x):
    return np.sum(x**2)

evg = EvolutionaryGradient(budget=1000, dim=10)
evg("func")