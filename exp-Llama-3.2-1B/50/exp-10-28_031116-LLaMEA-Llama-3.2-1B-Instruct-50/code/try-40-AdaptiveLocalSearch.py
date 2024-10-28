import numpy as np
import random

class AdaptiveLocalSearch:
    def __init__(self, budget, dim, alpha=0.45, mu=0.01, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.population = []
        self.fitnesses = []
        self.individuals = []
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        # Evaluate the function 100 times with adaptive sampling
        for _ in range(100):
            # Generate a random sample of size self.budget from the function's range
            sample = np.random.uniform(-self.sigma, self.sigma, self.dim)
            # Evaluate the function at the sample points
            fitnesses = [func(x) for x in sample]
            # Calculate the mean and standard deviation of the fitnesses
            mean_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)
            # Update the individual with the best fitness
            if mean_fitness < self.best_fitness:
                self.best_fitness = mean_fitness
                self.best_individual = sample
            # Refine the search space with local search
            if len(self.individuals) < self.budget:
                # Randomly select a new point within the current search space
                new_point = np.random.uniform(-self.sigma, self.sigma, self.dim)
                # Evaluate the new point
                fitness = func(new_point)
                # Update the individual with the new point
                self.individuals.append(new_point)
                # Update the fitness of the new point
                self.fitnesses.append(fitness)
            else:
                # Perform local search to refine the search space
                new_point = random.choice(self.individuals)
                # Evaluate the new point
                fitness = func(new_point)
                # Update the individual with the new point
                self.individuals.remove(new_point)
                self.individuals.append(new_point)
                # Update the fitness of the new point
                self.fitnesses.remove(fitness)
                self.fitnesses.append(fitness)
                # Update the best individual
                if mean_fitness < self.best_fitness:
                    self.best_fitness = mean_fitness
                    self.best_individual = new_point

    def get_best_individual(self):
        # Return the best individual found so far
        return self.best_individual

# Example usage:
# Create a new adaptive local search algorithm with 1000 function evaluations
al = AdaptiveLocalSearch(budget=1000, dim=10)
# Optimize the function f(x) = x^2 + 2x + 1 using the adaptive local search algorithm
func = lambda x: x**2 + 2*x + 1
best_individual = al.get_best_individual()
print("Best individual:", best_individual)
print("Best fitness:", func(best_individual))