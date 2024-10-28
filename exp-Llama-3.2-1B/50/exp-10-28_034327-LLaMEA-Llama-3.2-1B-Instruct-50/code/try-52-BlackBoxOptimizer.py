import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

class IteratedPermutationCooling(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Initialize the population with random points
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Iterate over the population
        for _ in range(self.iterations):
            # Select the best individual
            best_individual = self.population[np.argmax(self.population, axis=0)]

            # Generate a new population by iterated permutation
            new_population = self.iterated_permutation(self.population, best_individual)

            # Evaluate the new population
            new_evals = self.evaluate_fitness(new_population)

            # Update the population
            self.population = new_population
            self.func_evals = new_evals

        # Return the best individual in the final population
        return np.max(self.population, axis=0)

    def iterated_permutation(self, population, individual):
        # Generate a new population by iterated permutation
        new_population = population.copy()
        for _ in range(len(population) // 2):
            # Select two random individuals
            i, j = np.random.choice(len(population), 2, replace=False)

            # Swap the two individuals
            new_population[i], new_population[j] = new_population[j], new_population[i]

        return new_population