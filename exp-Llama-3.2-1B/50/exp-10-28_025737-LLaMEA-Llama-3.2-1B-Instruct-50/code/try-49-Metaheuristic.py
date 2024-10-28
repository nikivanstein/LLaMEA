import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize population with random individuals
        population = [self.__init__(self.budget, self.dim) for _ in range(100)]

        # Evaluate population and select fittest individuals
        fitnesses = [func(individual) for individual in population]
        fitnesses.sort(reverse=True)
        population = [individual for _, individual in zip(fitnesses, population)]

        # Refine the strategy
        while len(population) > 1:
            new_population = []
            for _ in range(self.budget):
                # Select a random parent from the current population
                parent1, parent2 = random.sample(population, 2)
                # Select a new individual by combining the two parents
                child = [x for x in parent1.search_space if x not in parent2.search_space]
                for i in range(self.dim):
                    if random.random() < 0.45:
                        child[i] += parent2.search_space[i]
                new_population.append(child)

            # Evaluate the new population and select the fittest individual
            fitnesses = [func(individual) for individual in new_population]
            fitnesses.sort(reverse=True)
            new_population = [individual for _, individual in zip(fitnesses, new_population)]

            # Update the population
            population = new_population

        # Return the fittest individual
        return population[0]

# Initialize the algorithm
algorithm = NovelMetaheuristicAlgorithm(100, 10)

# Evaluate the function
def evaluate_func(individual, budget):
    func_values = [func(x) for x in individual.search_space]
    fitnesses = [func(x) for x in func_values]
    fitnesses.sort(reverse=True)
    return fitnesses[0]

# Run the algorithm
best_individual = algorithm(algorithm, 100)

# Print the result
print("Best Individual:", best_individual)
print("Best Fitness:", evaluate_func(best_individual, 100))