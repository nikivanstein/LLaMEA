import numpy as np
import random

class ProbabilisticAdaptiveEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.score = self.evaluate_population()

    def initialize_population(self):
        return [self.sample_uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def sample_uniform(self, lower, upper, dim):
        return np.random.uniform(lower, upper, dim)

    def evaluate_population(self):
        scores = []
        for individual in self.population:
            score = self.func(individual)
            scores.append(score)
        return np.mean(scores)

    def __call__(self, func):
        for _ in range(self.budget):
            best_individual = self.population[np.argmax(self.score)]
            new_population = []
            for _ in range(self.population_size):
                if random.random() < 0.3:
                    new_individual = self.sample_uniform(-5.0, 5.0, self.dim)
                    new_population.append(new_individual)
                else:
                    new_individual = best_individual
                    new_population.append(new_individual)
            self.population = new_population
            self.score = self.evaluate_population()

    def select_solution(self):
        # Select the best individual in the population
        best_individual = self.population[np.argmax(self.score)]
        return best_individual

# Example usage
func = lambda x: x[0]**2 + x[1]**2
evolution = ProbabilisticAdaptiveEvolution(budget=100, dim=2)
evolution(func)
selected_solution = evolution.select_solution()
print(selected_solution)