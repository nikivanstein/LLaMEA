# Description: Evolutionary Black Box Optimization with Adaptive Mutation
# Code: 
# ```python
import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def fitness_evaluated(individual):
            return fitness(individual)

        def fitness_updated(individual):
            return fitness_evaluated(individual)

        def fitness_updated_new(individual):
            return fitness_evaluated(self.evaluate_fitness(individual))

        def fitness_updated_refined(individual):
            return fitness_evaluated(self.evaluate_fitness(new_individual))

        def fitness_updated_refined_new(individual):
            return fitness_evaluated(self.evaluate_fitness(refined_individual))

        def fitness_updated_refined_strong(individual):
            return fitness_evaluated(self.evaluate_fitness(new_individual))

        best_individual = self.population[np.argmax(fitness_scores)]
        new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
        new_individual = np.array(new_individual)
        if fitness(individual) > fitness_scores[np.argmax(fitness_scores)]:
            self.population[i] = new_individual
            self.fitness_scores[i] = fitness_evaluated(individual)
            if fitness_evaluated(individual) > fitness_updated(individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness_evaluated(individual)
            if fitness_evaluated(individual) > fitness_updated_new(individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness_evaluated(individual)
            if fitness_evaluated(individual) > fitness_updated_refined(individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness_evaluated(individual)
            if fitness_evaluated(individual) > fitness_updated_refined_new(individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness_evaluated(individual)
            if fitness_evaluated(individual) > fitness_updated_refined_strong(individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness_evaluated(individual)

        return self.population

    def mutate(self, individual):
        if random.random() < 0.01:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

algorithm_name = "EvolutionaryBlackBoxOptimizer"
description = "Novel Metaheuristic Algorithm for Black Box Optimization"