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

    def __call__(self, func, num_evals):
        # Initialize the population with random individuals
        population = [self.__call__(func) for _ in range(100)]

        # Evaluate the population and select the fittest individuals
        fitnesses = [func(individual) for individual in population]
        fitnesses.sort(key=fitnesses.index, reverse=True)
        population = [individual for index, individual in enumerate(population) if fitnesses[index] == fitnesses[0]]

        # Select the best individuals to mutate
        selected_individuals = random.sample(population, min(self.budget, len(population)))

        # Refine the strategy by changing the best individual
        for _ in range(100):
            # Select a random mutation point
            mutation_point = random.randint(0, self.dim - 1)

            # Create a new individual by changing the mutation point
            new_individual = [x if i!= mutation_point else x + 0.1 for i, x in enumerate(selected_individuals)]

            # Evaluate the new individual
            new_fitness = func(new_individual)

            # If the new individual is better than the current best individual, replace it
            if new_fitness > fitnesses[0]:
                selected_individuals.remove(selected_individuals[0])
                selected_individuals.append(new_individual)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in selected_individuals]

        return selected_individuals

# Test the algorithm
budget = 1000
dim = 10
func = np.sin
algorithm = NovelMetaheuristicAlgorithm(budget, dim)
selected_individuals = algorithm(__call__, budget)
print(selected_individuals)