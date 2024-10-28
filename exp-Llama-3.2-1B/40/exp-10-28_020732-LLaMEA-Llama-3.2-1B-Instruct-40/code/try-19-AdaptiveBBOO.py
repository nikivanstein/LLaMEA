import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.population = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def initialize_population(self, func, num_individuals):
        self.population = [func(np.random.uniform(self.search_space[0], self.search_space[1], self.dim)) for _ in range(num_individuals)]

    def mutate(self, individual):
        if np.random.rand() < 0.4:
            return individual + np.random.uniform(-1, 1, self.dim)
        return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.4:
            return np.concatenate((parent1[:np.random.randint(len(parent1))], parent2[np.random.randint(len(parent2))]))
        return np.concatenate((parent1, parent2))

    def evolve(self, population, func):
        for _ in range(self.budget):
            # Select the fittest individuals
            fittest_individuals = sorted(self.population, key=func, reverse=True)[:self.budget // 2]

            # Create a new population by crossover and mutation
            new_population = [self.crossover(parent, self.mutate(parent)) for parent in fittest_individuals]

            # Replace the old population with the new one
            self.population = new_population

            # Evaluate the new population
            self.func_evaluations.append([func(individual) for individual in new_population])

            # Update the best individual
            best_individual = max(self.func_evaluations, key=func)[0]
            best_individuals = [individual for individual in self.population if individual == best_individual]
            if len(best_individuals) > 0:
                best_individual = best_individuals[0]

        # Evaluate the best individual at the final search point
        best_individual = max(self.func_evaluations, key=func)[0]
        best_individuals = [individual for individual in self.population if individual == best_individual]
        if len(best_individuals) > 0:
            best_individual = best_individuals[0]

        # Return the best individual and its function value
        return best_individual, max(self.func_evaluations, key=func)[1]

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
best_individual, best_func_value = bboo.evolve(bboo.population, func)

# Plot the function
plt.plot(bboo.search_space, bboo.func_evaluations)
plt.plot(bboo.search_space, [best_func_value]*len(bboo.search_space), 'ro')
plt.show()