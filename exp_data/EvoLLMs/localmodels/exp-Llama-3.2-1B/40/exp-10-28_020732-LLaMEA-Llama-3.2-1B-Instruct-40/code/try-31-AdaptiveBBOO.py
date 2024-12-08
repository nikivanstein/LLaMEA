import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim, mutation_rate=0.01, crossover_rate=0.5, elite_size=10):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.population_size = 100
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite = None

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def initialize_population(self):
        # Initialize the population with random solutions
        population = [np.random.uniform(self.search_space) for _ in range(self.population_size)]
        return population

    def fitness(self, individual):
        # Evaluate the function at the individual
        func_value = func(individual)
        return func_value

    def select_elite(self, population, fitness):
        # Select the elite individuals with the highest fitness values
        elite = sorted(enumerate(fitness), key=lambda x: x[1], reverse=True)[:self.elite_size]
        return [population[i] for i in elite]

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = np.random.uniform(self.search_space, size=self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                child[i] = np.random.uniform(self.search_space)
        return child

    def mutate(self, individual):
        # Perform mutation on the individual
        if np.random.rand() < self.mutation_rate:
            index = np.random.randint(0, self.dim)
            individual[index] = np.random.uniform(self.search_space)
        return individual

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = self.mutate(x)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 5)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Update the elite individuals
elite = bboo.select_elite(bboo.population_size, bboo.fitness)
bboo.elite = elite

# Update the population with the elite individuals
new_population = bboo.initialize_population() + bboo.elite

# Optimize the function with the updated population
bboo.optimize(func)

# Update the elite individuals again
elite = bboo.select_elite(new_population, bboo.fitness)
bboo.elite = elite

# Update the population with the elite individuals
new_population = bboo.initialize_population() + bboo.elite

# Optimize the function with the updated population
bboo.optimize(func)

# Print the best solution and function value
print(f"Best solution: {bboo.elite[0]}, Best function value: {bboo.elite[1]}")

# Plot the function values
plt.plot(bboo.search_space, bboo.fitness(bboo.search_space))
plt.show()