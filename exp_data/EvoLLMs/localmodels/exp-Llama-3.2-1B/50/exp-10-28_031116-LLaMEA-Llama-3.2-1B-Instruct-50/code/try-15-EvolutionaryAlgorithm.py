import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.init_population()
        self.fitness_scores = []
        self.best_individual = None

    def init_population(self):
        # Initialize population with random individuals
        population = []
        for _ in range(100):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def __call__(self, func):
        # Evaluate function for each individual in the population
        fitness_scores = []
        for individual in self.population:
            func(individual)
            fitness = 1 / (func(individual) + 1)  # Avoid division by zero
            fitness_scores.append(fitness)
        # Select individuals with better fitness
        selected_indices = np.argsort(fitness_scores)[-self.budget:]
        selected_individuals = [self.population[i] for i in selected_indices]
        # Mutate selected individuals
        mutated_individuals = []
        for individual in selected_individuals:
            mutation_rate = 0.01
            for i in range(self.dim):
                if random.random() < mutation_rate:
                    mutation = random.uniform(-5.0, 5.0)
                    individual[i] += mutation
            mutated_individuals.append(individual)
        # Replace selected individuals with mutated ones
        self.population = mutated_individuals
        # Update fitness scores and best individual
        self.fitness_scores = fitness_scores
        self.best_individual = selected_individuals[0]

    def get_best_solution(self):
        return self.best_individual

# Example usage:
if __name__ == "__main__":
    # Set parameters
    budget = 1000
    dim = 10
    algorithm = EvolutionaryAlgorithm(budget, dim)

    # Define the black box function
    def func(individual):
        return individual @ individual

    # Run the algorithm
    best_solution = algorithm.get_best_solution()
    print("Best solution:", best_solution)
    print("Best fitness score:", algorithm.fitness_scores[-1])