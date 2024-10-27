import numpy as np
import random

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

    def mutate(self, individual):
        if random.random() < 0.2:
            bounds = bounds(individual)
            new_bound = bounds[random.randint(0, 1)]
            if new_bound[0] > -5.0:
                individual[new_bound[0]] = random.uniform(-5.0, new_bound[0])
            if new_bound[1] < 5.0:
                individual[new_bound[1]] = random.uniform(new_bound[1], 5.0)

class NNEO_BBOB(NNEO):
    def __init__(self, budget, dim, noise):
        super().__init__(budget, dim)
        self.noise = noise

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

        # Refine the solution
        for _ in range(int(self.budget * 0.1)):
            for i in range(self.population_size):
                x = self.population[i]
                bounds_x = bounds(x)
                noise_x = np.random.normal(0, 1, self.dim)
                new_x = x + noise_x
                bounds_new = bounds(new_x)
                if bounds_new[0] > -5.0:
                    new_x[new_bound[0]] = random.uniform(-5.0, new_bound[0])
                if bounds_new[1] < 5.0:
                    new_x[new_bound[1]] = random.uniform(bounds_new[1], 5.0)
                if new_x not in self.population:
                    self.population.append(new_x)

        return self.fitnesses

# Test the algorithm
func = lambda x: x**2
budget = 100
dim = 2
nneo = NNEO(budget, dim)
nneo_BBOB = NNEO_BBOB(budget, dim, noise=0.1)
nneo_BBOB(func)
print("NNEO:")
print(nneo())
print("NNEO_BBOB:")
print(nneo_BBOB())