import numpy as np

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

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, x, alpha):
        new_x = x.copy()
        for i in range(self.dim):
            new_x[i] += np.random.normal(0, 1) * alpha
            if new_x[i] < -5.0:
                new_x[i] = -5.0
            elif new_x[i] > 5.0:
                new_x[i] = 5.0
        return new_x

    def evaluate_fitness(self, new_individual):
        updated_individual = self.evaluate_fitness(new_individual)
        return updated_individual

# Example usage:
nneo = NNEO(100, 10)
func = lambda x: x**2
nneo(10, 10, func)
print(nneo(10, 10, func))