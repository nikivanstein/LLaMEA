import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def novel_iteration(self):
        # Initialize a new population of individuals
        new_population = self.evaluate_fitness(self.evaluate_fitness(np.random.uniform(-5.0, 5.0, self.dim), self))

        # Evaluate the new population
        new_population = np.array([self.evaluate_fitness(individual) for individual in new_population])

        # Select the fittest individuals
        fittest_individuals = np.argsort(new_population)[::-1][:self.budget]

        # Create new offspring by iterating over the selected individuals
        new_offspring = []
        for _ in range(self.budget):
            parent1, parent2 = fittest_individuals[_//self.budget], fittest_individuals[((_//self.budget)+1)%self.budget]
            child = (parent1 + parent2) / 2
            new_offspring.append(child)

        # Update the population
        self.new_population = new_population + new_offspring

        # Cool the temperature
        self.temperature = min(1.0, self.temperature - 0.01)

        return new_population

    def evaluate_fitness(self, func, population):
        results = [func(individual) for individual in population]
        return np.array(results)

    def optimize_function(self, func, population_size, budget, dim):
        new_population = self.novel_iteration()
        best_individual = new_population[np.argmax([self.evaluate_fitness(func, individual) for individual in new_population])]

        # Update the population and the best individual
        self.new_population = new_population
        self.best_individual = best_individual
        self.func_evals = 0
        return self.best_individual