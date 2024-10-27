import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def novel_metaheuristic_algorithm(self, func, budget, dim, mutation_rate, cooling_rate, exploration_rate):
        # Initialize the current best point and fitness
        current_best_point = self.search_space[0], self.search_space[1]
        current_fitness = func(current_best_point)
        # Initialize the population
        population = [current_best_point] * 100
        # Run the differential evolution algorithm
        for _ in range(budget):
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]
            # Select the fittest individuals to reproduce
            fittest_individuals = np.argsort(fitnesses)[-10:]
            # Select a random subset of individuals to mutate
            mutated_individuals = random.sample(fittest_individuals, 50)
            # Mutate the selected individuals
            mutated_individuals = [self.mutate(individual, mutation_rate, exploration_rate) for individual in mutated_individuals]
            # Evaluate the fitness of the mutated individuals
            mutated_fitnesses = [func(individual) for individual in mutated_individuals]
            # Select the fittest mutated individuals to reproduce
            fittest_mutated_individuals = np.argsort(mutated_fitnesses)[-10:]
            # Replace the least fit individuals with the fittest mutated individuals
            population = mutated_individuals + fittest_mutated_individuals
            # Update the current best point and fitness
            current_best_point = population[np.argmin(fitnesses)]
            current_fitness = func(current_best_point)
            # If the current best point has a lower fitness, update the current best point
            if current_fitness < current_best_fitness:
                current_best_point = current_best_point
                current_best_fitness = current_fitness
            # If the current best point has a lower fitness, update the current best point
            if current_fitness < self.current_best_fitness:
                self.current_best_fitness = current_fitness
        # Return the current best point and fitness
        return current_best_point, current_fitness

    def mutate(self, individual, mutation_rate, exploration_rate):
        # Generate a new individual by perturbing the current individual
        new_individual = individual.copy()
        # Perturb the current individual
        for _ in range(self.dim):
            if random.random() < mutation_rate:
                new_individual[_] += random.uniform(-1, 1)
                if new_individual[_] < -5.0:
                    new_individual[_] = -5.0
                elif new_individual[_] > 5.0:
                    new_individual[_] = 5.0
        # Increase the fitness of the new individual
        new_fitness = func(new_individual)
        # If the new fitness is better, update the new individual
        if new_fitness < self.current_best_fitness:
            self.current_best_fitness = new_fitness
        return new_individual

    def current_best_fitness(self):
        return self.current_best_fitness

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 