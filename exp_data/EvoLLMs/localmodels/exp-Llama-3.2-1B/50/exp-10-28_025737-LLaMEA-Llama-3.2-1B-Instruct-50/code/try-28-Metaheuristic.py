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
        # Initialize the population with random individuals
        population = [self.__init__(self.budget, self.dim) for _ in range(100)]

        # Run the selection process
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [self.__call__(func) for func in population]

            # Select the fittest individuals
            parents = [population[np.argmax(fitness)]]
            for _ in range(99):
                # Select two parents at random
                parent1, parent2 = random.sample(parents, 2)

                # Create a new child by crossover and mutation
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                # Replace the parents with the new child
                parents = [child]

            # Replace the population with the new individuals
            population = parents

        # Return the fittest individual
        return population[np.argmax(fitness)]

def crossover(parent1, parent2):
    # Select a random crossover point
    crossover_point = random.randint(1, self.dim - 1)

    # Split the parents into two halves
    child1 = parent1[:crossover_point]
    child2 = parent2[crossover_point:]

    # Combine the two halves
    child = child1 + child2

    return child

def mutate(individual):
    # Select a random mutation point
    mutation_point = random.randint(0, self.dim - 1)

    # Swap the two points
    individual[mutation_point], individual[mutation_point + 1] = individual[mutation_point + 1], individual[mutation_point]

    return individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# Novel Metaheuristic Algorithm for Black Box Optimization
# ```