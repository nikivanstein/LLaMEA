import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, iterations=1000, mutation_rate=0.01, cooling_rate=0.99):
        # Initialize the population with random individuals
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        for _ in range(iterations):
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals
            fittest_individuals = np.array(population)[fitness.argsort()[::-1][:self.budget]]

            # Generate new individuals using the fittest individuals
            new_individuals = []
            for _ in range(self.budget - len(fittest_individuals)):
                # Select a random individual from the fittest individuals
                parent1 = fittest_individuals[np.random.randint(0, fittest_individuals.shape[0])]
                parent2 = fittest_individuals[np.random.randint(0, fittest_individuals.shape[0])]
                # Crossover (random walk)
                child = random.uniform(parent1, parent2) if random.random() < mutation_rate else parent1 + random.uniform(-1, 1) * random.random() * (parent2 - parent1)
                # Mutation
                if random.random() < mutation_rate:
                    child = random.uniform(parent1, parent2) if random.random() < mutation_rate else parent1 + random.uniform(-1, 1) * random.random() * (parent2 - parent1)
                new_individuals.append(child)

            # Replace the old population with the new individuals
            population = new_individuals

        # Evaluate the fitness of the best individual
        best_individual = np.argmax(fitness)
        best_fitness = fitness[best_individual]
        # Return the best individual and its fitness
        return best_individual, best_fitness

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 