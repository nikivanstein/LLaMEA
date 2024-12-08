import numpy as np
import random

class PAEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def evaluate(self, func):
        scores = []
        for individual in self.population:
            score = func(individual)
            scores.append(score)
        return scores

    def selection(self, scores):
        probabilities = np.array(scores) / np.sum(scores)
        selected_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities)
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def crossover(self, individuals1, individuals2):
        offspring = []
        for i in range(len(individuals1)):
            if random.random() < self.crossover_rate:
                parent1 = individuals1[i]
                parent2 = individuals2[i]
                child = [(parent1[j] + parent2[j]) / 2 for j in range(self.dim)]
                offspring.append(child)
            else:
                offspring.append(parent1)
        return offspring

    def mutation(self, individuals):
        mutated_individuals = []
        for individual in individuals:
            mutated_individual = individual.copy()
            for j in range(self.dim):
                if random.random() < self.mutation_rate:
                    mutated_individual[j] += random.uniform(-1.0, 1.0)
                    mutated_individual[j] = max(-5.0, min(5.0, mutated_individual[j]))
            mutated_individuals.append(mutated_individual)
        return mutated_individuals

    def optimize(self, func):
        for _ in range(self.budget):
            scores = self.evaluate(func)
            selected_individuals = self.selection(scores)
            offspring = self.crossover(selected_individuals, selected_individuals)
            mutated_offspring = self.mutation(offspring)
            self.population = mutated_offspring
            scores = self.evaluate(func)
            selected_indices = np.argmax(scores)
            best_individual = self.population[selected_indices]
            print(f"Best individual: {best_individual}")
            print(f"Best score: {scores[selected_indices]}")