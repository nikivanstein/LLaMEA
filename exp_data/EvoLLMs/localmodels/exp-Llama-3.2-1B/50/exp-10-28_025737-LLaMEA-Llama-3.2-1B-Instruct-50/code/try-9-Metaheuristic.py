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

    def __call__(self, func, population_size=100, mutation_rate=0.1, num_generations=100):
        # Initialize the population
        population = [self.__call__(func) for _ in range(population_size)]

        # Evaluate the population
        fitnesses = [self.evaluate_fitness(individual, func, population_size) for individual, func in zip(population, func)]

        # Select the fittest individuals
        fittest_indices = np.argsort(fitnesses)[::-1][:self.budget]

        # Refine the strategy
        for _ in range(num_generations):
            new_population = []
            for _ in range(population_size):
                # Select two parents
                parent1, parent2 = random.sample(fittest_indices, 2)

                # Create a new individual
                child = self.__call__(func, parent1, parent2, mutation_rate)

                # Add the child to the new population
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

        # Evaluate the final population
        fitnesses = [self.evaluate_fitness(individual, func, population_size) for individual, func in zip(population, func)]

        # Select the fittest individuals
        fittest_indices = np.argsort(fitnesses)[::-1][:self.budget]

        return fittest_indices

def evaluate_fitness(individual, func, population_size):
    # Evaluate the function a limited number of times
    num_evals = min(population_size, len(func(individual)))
    func_values = [func(x) for x in random.sample(func(individual), num_evals)]

    # Select the best function value
    best_func = max(set(func_values), key=func_values.count)

    # Update the search space
    individual = [x for x in individual if x not in best_func]

    return individual

# Example usage
func = lambda x: x**2
algorithm = NovelMetaheuristicAlgorithm(10, 5)
fittest_indices = algorithm(__call__, func, mutation_rate=0.01, num_generations=100)
print(fittest_indices)