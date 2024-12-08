import numpy as np

class ProbabilisticMutationCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.population = np.random.uniform(self.search_space[0], self.search_space[1], (self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)
        self.best_solution = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim,))

    def __call__(self, func):
        # Evaluate the fitness of the current population
        for i in range(self.population_size):
            self.fitness_values[i] = func(self.population[i])

        # Select the best solution
        idx = np.argmin(self.fitness_values)
        self.best_solution = self.population[idx]

        # Perform mutation and crossover
        for i in range(self.population_size):
            if np.random.rand() < 0.3:
                # Mutation: perturb the solution by a small amount
                self.population[i] += np.random.uniform(-0.1, 0.1, self.dim)
                self.population[i] = np.clip(self.population[i], self.search_space[0], self.search_space[1])

            if np.random.rand() < 0.3:
                # Crossover: combine two solutions
                parent1_idx = np.random.randint(0, self.population_size)
                parent2_idx = np.random.randint(0, self.population_size)
                child = self.population[parent1_idx] + (self.population[parent2_idx] - self.population[parent1_idx])
                self.population[i] = np.clip(child, self.search_space[0], self.search_space[1])

        # Evaluate the fitness of the new population
        for i in range(self.population_size):
            self.fitness_values[i] = func(self.population[i])

        # Replace the least fit solution with the best solution
        idx = np.argmin(self.fitness_values)
        self.population[idx] = self.best_solution

        # Check if the budget is exhausted
        if self.fitness_values[np.argmin(self.fitness_values)] < func(self.best_solution):
            return
        if self.budget == 0:
            raise Exception("Budget exhausted")

        # Decrement the budget
        self.budget -= 1

# Example usage
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = ProbabilisticMutationCrossover(budget, dim)
optimizer(func)