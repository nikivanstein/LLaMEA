import numpy as np

class EvolutionaryStrategy:
    def __init__(self, budget, dim, mutation_rate=0.1, bounds_clip=5.0):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-bounds_clip, bounds_clip, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.mutation_rate = mutation_rate
        self.bounds_clip = bounds_clip

    def __call__(self, func, problem):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min(), x.max())

        def fitness(x):
            return objective(x)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness_value = fitness(x)
                if fitness_value < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness_value
                    self.population[i] = x

            # Select new individuals based on bounds clipping and mutation
            new_individuals = []
            for _ in range(self.population_size):
                x = self.population[i]
                if np.random.rand() < self.mutation_rate:
                    # Randomly change one element in the individual
                    idx = np.random.randint(0, self.dim)
                    x[idx] = np.clip(x[idx] + np.random.uniform(-1.0, 1.0), self.bounds_clip, self.bounds_clip)
                new_individuals.append(x)

            self.population = np.array(new_individuals)

        return self.fitnesses

# Example usage:
problem = "BBOB"
func = "example_func"
best_solution = EvolutionaryStrategy(budget=100, dim=10).__call__(func, problem)
print(best_solution)