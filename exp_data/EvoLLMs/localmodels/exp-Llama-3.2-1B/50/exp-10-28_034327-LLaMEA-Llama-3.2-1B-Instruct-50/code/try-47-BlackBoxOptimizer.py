import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = None
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        # Initialize the population with random individuals
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        while self.func_evals < self.budget:
            # Generate a new population by iterated permutation
            self.population = np.array([self.iterated_permutation(self.population[i], self.mutation_rate) for i in range(self.population_size)])

            # Evaluate the function for each individual in the population
            self.func_evals = 0
            for individual in self.population:
                value = func(individual)
                if -5.0 <= individual[0] <= 5.0 and -5.0 <= individual[1] <= 5.0:
                    self.func_evals += 1
                    return value

        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation(self, individual, mutation_rate):
        # Create a copy of the individual
        new_individual = individual.copy()

        # Randomly swap two elements in the individual
        index1, index2 = np.random.choice([0, 1], size=2, replace=False)
        new_individual[index1], new_individual[index2] = new_individual[index2], new_individual[index1]

        # Apply the mutation rate to the new individual
        if np.random.rand() < mutation_rate:
            new_individual[index1], new_individual[index2] = new_individual[index2], new_individual[index1]

        return new_individual