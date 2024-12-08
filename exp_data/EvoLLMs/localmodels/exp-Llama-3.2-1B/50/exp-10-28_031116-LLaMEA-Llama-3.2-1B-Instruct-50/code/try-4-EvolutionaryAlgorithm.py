import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_populations=10, num_generations=100, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.num_populations = num_populations
        self.num_generations = num_generations
        self.mutation_prob = mutation_prob
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.num_populations)]

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        # Select the best individual from each population
        selected_individuals = []
        for _ in range(self.num_populations):
            individual = random.choice(self.population)
            best_individual = None
            best_value = evaluate_func(individual)
            for i, x in enumerate(individual):
                value = evaluate_func(x)
                if value < best_value:
                    best_individual = individual
                    best_value = value
            selected_individuals.append(best_individual)

        # Evaluate the selected individuals
        selected_values = [evaluate_func(individual) for individual in selected_individuals]

        # Select the best individual based on the budget
        selected_index = np.argsort(-selected_values)[:self.budget]
        selected_individuals = [selected_individuals[i] for i in selected_index]

        # Select a random individual from the remaining population
        remaining_population = [individual for individual in self.population if individual not in selected_individuals]
        remaining_index = random.choices(range(len(remaining_population)), weights=remaining_values, k=1)[0]
        selected_individuals.append(remaining_population[remaining_index])

        # Create a new population by replacing the selected individuals with new ones
        new_population = []
        for _ in range(self.num_populations):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            best_value = evaluate_func(individual)
            for i, x in enumerate(individual):
                value = evaluate_func(x)
                if value < best_value:
                    best_individual = individual
                    best_value = value
            new_population.append(best_individual)

        # Replace the old population with the new one
        self.population = new_population

        # Evaluate the new population
        new_values = [evaluate_func(individual) for individual in new_population]

        # Select the best individual based on the budget
        new_selected_index = np.argsort(-new_values)[:self.budget]
        new_selected_individuals = [new_population[i] for i in new_selected_index]

        # Select a random individual from the remaining population
        remaining_population = [individual for individual in self.population if individual not in new_selected_individuals]
        remaining_index = random.choices(range(len(remaining_population)), weights=remaining_values, k=1)[0]
        new_selected_individuals.append(remaining_population[remaining_index])

        # Create a new population by replacing the selected individuals with new ones
        new_new_population = []
        for _ in range(self.num_populations):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            best_value = evaluate_func(individual)
            for i, x in enumerate(individual):
                value = evaluate_func(x)
                if value < best_value:
                    best_individual = individual
                    best_value = value
            new_new_population.append(best_individual)

        # Replace the old population with the new one
        self.population = new_new_population

        return selected_individuals

    def fitness(self, func, individual):
        return evaluate_func(individual)

    def mutate(self, individual):
        if random.random() < self.mutation_prob:
            index1, index2 = random.sample(range(self.dim), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
            return individual