import numpy as np
import random
import copy

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the temperature
            self.temp = max(0.1, self.temp - 0.01)

            num_evals += 1

        return self.best_func

class EvolutionaryOptimizationMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.population = []
        self.population_history = []

    def __call__(self, func, population_size=100):
        for _ in range(population_size):
            # Generate a random population
            population = [copy.deepcopy(func) for _ in range(population_size)]

            # Evaluate the population
            fitnesses = self.evaluate_fitness(population)

            # Select the fittest individuals
            self.population = self.select_fittest(population, fitnesses)

            # Optimize the fittest individuals
            self.population_history.append(self.population)

            # Optimize the fittest individual
            self.population = self.optimize_fittest_individual(population)

            # Update the best function
            self.best_func = self.population[0]

            # Update the temperature
            self.temp = max(0.1, self.temp - 0.01)

            # Check if the optimization is complete
            if np.random.rand() < 0.25:
                break

        return self.best_func

    def evaluate_fitness(self, population):
        # Evaluate the fitness of each individual in the population
        fitnesses = []
        for individual in population:
            # Evaluate the function using the given function
            func = self.evaluate_function(individual)
            fitnesses.append(func)

        return fitnesses

    def select_fittest(self, population, fitnesses):
        # Select the fittest individuals based on their fitness
        fittest_individuals = []
        for fitness in fitnesses:
            if fitness == max(fitnesses):
                fittest_individuals.append(population[fitnesses.index(max(fitnesses))])
            else:
                fittest_individuals.append(population[np.argmin(fitnesses)])

        return fittest_individuals

    def optimize_fittest_individual(self, population):
        # Optimize the fittest individual using a genetic algorithm
        # This is a simple example and may need to be modified based on the specific problem
        fitnesses = self.evaluate_fitness(population)
        fittest_individuals = self.select_fittest(population, fitnesses)
        # Optimize the fittest individual using a genetic algorithm
        # This is a simple example and may need to be modified based on the specific problem
        return fittest_individuals[0]

    def evaluate_function(self, individual):
        # Evaluate the function using the given function
        func = self.evaluate_function(individual)
        return func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 