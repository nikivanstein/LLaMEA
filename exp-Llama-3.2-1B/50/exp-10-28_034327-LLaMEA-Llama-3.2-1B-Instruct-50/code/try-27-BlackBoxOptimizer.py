import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = None
        self.population_fitness = None

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

    def initialize_population(self, dim):
        self.population = np.random.uniform(-5.0, 5.0, dim)
        self.population_fitness = np.array([func(self.population) for func in BlackBoxOptimizer.__call__])

    def mutate(self, point, mutation_rate):
        if np.random.rand() < mutation_rate:
            self.population[point] = np.random.uniform(-5.0, 5.0)

    def __next_generation(self, new_population):
        # Select the fittest individuals
        fittest_indices = np.argsort(self.population_fitness)
        new_population = new_population[fittest_indices]

        # Generate a new population by iterated permutation and cooling
        for _ in range(self.budget // 2):
            new_point = np.random.uniform(-5.0, 5.0, self.dim)
            new_point = self.iterated_permutation(new_point, self.population)
            new_population.append(new_point)

        return new_population

    def iterated_permutation(self, point, population):
        # Apply iterated permutation
        new_point = point
        while len(population) > 0 and new_point in population:
            new_point = self.cooling(new_point, population)

        return new_point

    def cooling(self, point, population):
        # Apply cooling
        new_point = point
        for _ in range(self.dim):
            r = np.random.uniform(0, 1)
            if r < 0.5:
                new_point -= 0.1 * (point - new_point)
            else:
                new_point += 0.1 * (point - new_point)
        return new_point