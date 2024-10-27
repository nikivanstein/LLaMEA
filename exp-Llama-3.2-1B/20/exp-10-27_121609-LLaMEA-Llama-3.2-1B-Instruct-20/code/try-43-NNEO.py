import numpy as np

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if np.random.rand() < 0.2:
                bounds = bounds[individual]
                bounds[0] = min(bounds[0] - 1.0, -5.0)
                bounds[1] = max(bounds[1] + 1.0, 5.0)
            return individual

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return individual

        for _ in range(self.budget):
            new_individual = evaluate_fitness(self.population)
            if new_individual not in self.population_history:
                self.population_history.append(new_individual)
                self.population = np.array([mutate(individual) for individual in self.population])

        return self.fitnesses

# Example usage:
# 
# Create a new NNEO instance with a budget of 1000 and a dimension of 10
nneo = NNEO(1000, 10)

# Optimize the function f(x) = x^2 + 2x + 1 using the NNEO algorithm
nneo_func = lambda x: x**2 + 2*x + 1
nneo.optimize(f=nneo_func, func=functools.lambdify(np.linspace(-5.0, 5.0, 1000), nneo_func))

# Print the fitnesses
print(nneo.fitnesses)