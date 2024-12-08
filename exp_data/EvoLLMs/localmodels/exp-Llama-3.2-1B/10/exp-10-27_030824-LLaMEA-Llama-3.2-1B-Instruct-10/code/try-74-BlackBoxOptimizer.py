import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def novel_metaheuristic(self, func, budget, dim):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Initialize the population with the best individual
        best_individual = population[0]
        best_fitness = func(best_individual)

        # Run the metaheuristic for the specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals for the next generation
            selected_indices = np.argsort(fitnesses)[-budget:]
            selected_individuals = [population[i] for i in selected_indices]

            # Create a new generation by linearly interpolating between the selected individuals
            new_population = []
            for _ in range(100):
                # Randomly select two parents from the selected individuals
                parent1, parent2 = random.sample(selected_individuals, 2)

                # Linearly interpolate between the parents to create a new individual
                new_individual = (1 - random.random()) * parent1 + random.random() * parent2

                # Add the new individual to the new population
                new_population.append(new_individual)

            # Replace the old population with the new population
            population = new_population

            # Update the best individual and its fitness
            best_individual = np.min(population, axis=0)
            best_fitness = func(best_individual)

        # Return the best individual and its fitness
        return best_individual, best_fitness


# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Code: 