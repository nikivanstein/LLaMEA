import numpy as np

class NonUniformSampling:
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

        def sample(self):
            indices = np.random.choice(self.population_size, self.budget, replace=False)
            return self.population[indices]

        def mutate(self, individual):
            # Non-Uniform Sampling Mutation
            bounds = bounds(individual)
            for i in range(self.dim):
                x = individual
                if np.random.rand() < 0.2:
                    x[i] += np.random.uniform(-0.1, 0.1)
                if np.random.rand() < 0.2:
                    x[i] -= np.random.uniform(-0.1, 0.1)
            return individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.sample()
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

# One-line description
# NonUniformSampling: A novel evolutionary algorithm that uses non-uniform sampling to refine the solution strategy.

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

        def sample(self):
            indices = np.random.choice(self.population_size, self.budget, replace=False)
            return self.population[indices]

        def mutate(self, individual):
            # Non-UniformSampling Mutation
            bounds = bounds(individual)
            for i in range(self.dim):
                x = individual
                if np.random.rand() < 0.2:
                    x[i] += np.random.uniform(-0.1, 0.1)
                if np.random.rand() < 0.2:
                    x[i] -= np.random.uniform(-0.1, 0.1)
            return individual

        def __next__(self):
            new_individual = sample()
            return mutate(new_individual)

        return self.__next__

# Example usage
problem = NNEO(budget=100, dim=5)
func = lambda x: x**2
solution = problem(problem(func))
print(solution)