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

        # Evolve the population for a fixed number of generations
        for _ in range(100):
            # Evaluate the function for each individual
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for individual, fitness in zip(population, fitnesses) if fitness == max(fitnesses)]

            # Create new individuals by refining the fittest individuals
            new_individuals = []
            for _ in range(self.budget):
                # Select two parents from the fittest individuals
                parent1, parent2 = random.sample(fittest_individuals, 2)

                # Refine the parents to produce a new individual
                child = self.refine(parent1, parent2)

                # Add the child to the new individuals list
                new_individuals.append(child)

            # Replace the old population with the new individuals
            population = new_individuals

        # Evaluate the function for each individual
        fitnesses = [func(individual) for individual in population]

        # Select the best individual
        best_individual = max(set(fitnesses), key=fitnesses.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_individual]

        return best_individual

    def refine(self, individual1, individual2):
        # Calculate the fitness difference between the two individuals
        fitness_diff = abs(individual1 - individual2)

        # Refine the individual with the lower fitness
        if fitness_diff < 0.45:
            return individual1
        else:
            return individual2

# Initialize the Metaheuristic Algorithm
algorithm = NovelMetaheuristicAlgorithm(100, 10)

# Test the algorithm
func = lambda x: np.sin(x)
best_individual = algorithm(func)
print("Best Individual:", best_individual)
print("Best Fitness:", func(best_individual))