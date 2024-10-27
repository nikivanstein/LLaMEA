import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func, algorithm='Hyperband'):
        if algorithm == 'Hyperband':
            return self.hyperband(func)
        elif algorithm == 'Bayesian':
            return self.bayesian(func)
        else:
            raise ValueError("Invalid algorithm. Please choose 'Hyperband' or 'Bayesian'.")

    def hyperband(self, func):
        # Initialize the population with random individuals
        population = self.initialize_population(self.budget, self.dim)

        # Run the Hyperband algorithm
        for _ in range(10):  # Run for 10 iterations
            # Evaluate the fitness of each individual
            fitnesses = [self.evaluate_fitness(individual, func) for individual in population]
            # Select the fittest individuals
            selected_individuals = np.argsort(fitnesses)[-self.budget:]
            # Create a new population by combining the selected individuals
            population = [self.combine_individuals(population, selected_individuals) for _ in range(self.dim)]
            # Update the search space
            self.search_space = (min(self.search_space[0], population[0]), max(self.search_space[1], population[-1]))
        # Evaluate the fitness of the final population
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]
        # Select the fittest individual
        selected_individual = np.argsort(fitnesses)[-1]
        # Return the fittest individual
        return selected_individual

    def bayesian(self, func):
        # Initialize the population with random individuals
        population = self.initialize_population(self.budget, self.dim)

        # Run the Bayesian optimization algorithm
        for _ in range(10):  # Run for 10 iterations
            # Evaluate the fitness of each individual
            fitnesses = [self.evaluate_fitness(individual, func) for individual in population]
            # Select the fittest individuals
            selected_individuals = np.argsort(fitnesses)[-self.budget:]
            # Create a new population by combining the selected individuals
            population = [self.combine_individuals(population, selected_individuals) for _ in range(self.dim)]
            # Update the search space
            self.search_space = (min(self.search_space[0], population[0]), max(self.search_space[1], population[-1]))
        # Evaluate the fitness of the final population
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]
        # Select the fittest individual
        selected_individual = np.argsort(fitnesses)[-1]
        # Return the fittest individual
        return selected_individual

    def initialize_population(self, budget, dim):
        # Initialize the population with random individuals
        population = np.random.uniform(self.search_space[0], self.search_space[1], (budget, dim))
        return population

    def evaluate_fitness(self, individual, func):
        # Evaluate the fitness of an individual
        fitness = func(individual)
        return fitness

    def combine_individuals(self, population, selected_individuals):
        # Combine the selected individuals into a new population
        new_population = np.zeros((len(selected_individuals), self.dim))
        for i, selected_individual in enumerate(selected_individuals):
            new_population[i] = population[i]
        for individual in population:
            if individual not in selected_individuals:
                new_population[i] = individual
        return new_population

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()