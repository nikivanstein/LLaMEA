import numpy as np

class AdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func, logger):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x
                    self.population_history.append((i, x))

            # Select the best solution based on a probability of 0.2
            selected_indices = np.random.choice(self.population_size, size=self.population_size, replace=False, p=[0.8, 0.2])
            selected_indices = np.array(selected_indices)

            # Create a new population by evolving the selected solutions
            new_population = np.zeros((self.population_size, self.dim))
            for j in range(self.population_size):
                new_individual = self.evaluate_fitness(new_individual)
                new_population[j] = new_individual

            # Replace the old population with the new one
            self.population = np.vstack((self.population, new_population))

        return self.fitnesses

    def evaluate_fitness(self, individual):
        # Use the budget to evaluate the function
        # For simplicity, assume the function is evaluated in a single call
        fitness = objective(individual)
        self.fitnesses[self.population_history.index((self.population_size, individual))] = fitness
        return fitness