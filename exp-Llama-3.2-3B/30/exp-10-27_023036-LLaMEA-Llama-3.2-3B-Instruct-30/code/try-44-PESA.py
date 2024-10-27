import numpy as np

class PESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)
        self.best_solution = self.population[np.argmax(self.fitness_values)]

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness_values = func(self.population)

            # Replace the worst solution with a new one
            self.population[np.argmin(self.fitness_values)] = self._mutate(self.population[np.argmin(self.fitness_values)])

            # Select the best solution
            self.best_solution = self.population[np.argmax(self.fitness_values)]

            # Refine the strategy with 0.3 probability
            if np.random.rand() < 0.3:
                self.population[np.random.choice(self.population_size)] = self._crossover(self.population[np.random.choice(self.population_size)], self.best_solution)

    def _mutate(self, solution):
        # Perform mutation with a probability of 0.1
        if np.random.rand() < 0.1:
            return solution + np.random.uniform(-1.0, 1.0, self.dim)
        else:
            return solution

    def _crossover(self, parent1, parent2):
        # Perform crossover with a probability of 0.5
        if np.random.rand() < 0.5:
            child = np.random.choice([parent1, parent2])
            return child + np.random.uniform(-1.0, 1.0, self.dim)
        else:
            return parent1