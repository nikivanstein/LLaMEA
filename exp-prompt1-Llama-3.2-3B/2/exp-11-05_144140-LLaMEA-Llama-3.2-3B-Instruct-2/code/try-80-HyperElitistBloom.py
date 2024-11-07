import random
import numpy as np

class HyperElitistBloom:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.elitist = None
        self.population = self.initialize_population()
        self.learning_rate = 0.1
        self.ranking_weights = np.array([1 / np.arange(1, self.budget + 1) for _ in range(self.budget)])

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        # Ensure that the first 10% of the population are unique
        unique_indices = random.sample(range(self.budget), int(self.budget * 0.1))
        for i in unique_indices:
            population[i] = np.random.uniform(-5.0, 5.0, self.dim)
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            # Select the best 20% of the population
            selected = self.population[np.argsort(func(self.population))[:int(0.2 * self.budget)]]
            # Select the best individual from the selected population
            best = selected[np.argmin(func(selected))]
            # Add the best individual to the elitist list
            if self.elitist is None or np.linalg.norm(best - self.elitist) > 1e-6:
                self.elitist = best
            # Replace the worst individual with the new best individual
            self.population[self.budget - 1] = best
            # Replace the worst individual in the selected population with a new random individual
            worst = selected[np.argmax(func(selected))]
            if random.random() < 0.02:  # 2% chance of mutation
                # Introduce a new mutation strategy: use a Gaussian mutation
                self.population[np.argmin(func(self.population))] = best + np.random.normal(0, 0.1 * self.learning_rate, self.dim)
                self.learning_rate *= 0.9
            # Apply a new "Bloom" selection strategy: use a weighted selection
            weights = np.array([func(individual) for individual in selected])
            weights /= np.sum(weights)
            new_population = []
            for _ in range(self.budget):
                r = random.random()
                cumulative_weight = 0
                for individual in selected:
                    cumulative_weight += weights[individual]
                    if cumulative_weight >= r:
                        new_population.append(individual)
                        break
            # Use a more efficient ranking-based selection
            if random.random() < 0.01:  # 1% chance of using ranking-based selection
                ranks = np.argsort(func(self.population)) + 1
                self.ranking_weights = np.array([1 / rank for rank in ranks])
                self.ranking_weights /= np.sum(self.ranking_weights)
                new_population = []
                for _ in range(self.budget):
                    r = random.random()
                    cumulative_weight = 0
                    for individual in selected:
                        cumulative_weight += self.ranking_weights[individual]
                        if cumulative_weight >= r:
                            new_population.append(individual)
                            break
            self.population = new_population

    def get_elitist(self):
        return self.elitist
