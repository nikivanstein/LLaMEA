import numpy as np

class AdaptiveMutation(NNEO):
    def __init__(self, budget, dim, mutation_rate):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if np.random.rand() < self.mutation_rate:
                i, j = np.random.randint(0, self.dim), np.random.randint(0, self.dim)
                x[i], x[j] = x[j], x[i]
            return x

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = mutate(x)

        return self.fitnesses