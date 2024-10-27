import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        if np.random.rand() < 0.05:
            i, j = np.random.choice(self.dim, 2, replace=False)
            self.search_space[i], self.search_space[j] = self.search_space[j], self.search_space[i]
        return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.05:
            i, j = np.random.choice(self.dim, 2, replace=False)
            parent1[i], parent1[j] = parent1[j], parent1[i]
        return parent1

    def selection(self, population):
        return np.array(population[np.argsort(self.func_evaluations):])

    def initialize(self, population):
        return np.array(population)

# Example usage
budget = 100
dim = 10
problem = NovelMetaheuristic(budget, dim)

# Initialize the population
population = np.random.uniform(-5.0, 5.0, (dim, 10)).astype(np.float64)

# Run the algorithm
solution = NovelMetaheuristic(budget, dim).initialize(population)
best_solution = solution

# Evaluate the function
best_func_value = np.nan
for _ in range(self.budget):
    func_value = NovelMetaheuristic(budget, dim).__call__(best_solution)
    if np.isnan(func_value) or np.isinf(func_value):
        raise ValueError("Invalid function value")
    if func_value < 0 or func_value > 1:
        raise ValueError("Function value must be between 0 and 1")
    best_func_value = np.nan
    if func_value > best_func_value:
        best_func_value = func_value

# Update the solution
new_individual = NovelMetaheuristic(budget, dim).evaluate_fitness(best_solution)
best_solution = np.array([new_individual]).astype(np.float64)
best_func_value = np.nan
for _ in range(self.budget):
    func_value = NovelMetaheuristic(budget, dim).__call__(best_solution)
    if np.isnan(func_value) or np.isinf(func_value):
        raise ValueError("Invalid function value")
    if func_value < 0 or func_value > 1:
        raise ValueError("Function value must be between 0 and 1")
    best_func_value = np.nan
    if func_value > best_func_value:
        best_func_value = func_value

# Update the population
population = NovelMetaheuristic(budget, dim).selection(population)
population = NovelMetaheuristic(budget, dim).initialize(population)

# Evaluate the function
best_func_value = np.nan
for _ in range(self.budget):
    func_value = NovelMetaheuristic(budget, dim).__call__(best_solution)
    if np.isnan(func_value) or np.isinf(func_value):
        raise ValueError("Invalid function value")
    if func_value < 0 or func_value > 1:
        raise ValueError("Function value must be between 0 and 1")
    best_func_value = np.nan
    if func_value > best_func_value:
        best_func_value = func_value

# Print the results
print("Best solution:", best_solution)
print("Best function value:", best_func_value)