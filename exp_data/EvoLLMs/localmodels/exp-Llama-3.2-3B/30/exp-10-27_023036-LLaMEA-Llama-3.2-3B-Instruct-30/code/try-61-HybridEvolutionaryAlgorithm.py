import numpy as np
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.best_individual = np.random.uniform(-5.0, 5.0, size=self.dim)
        self.best_score = float('inf')

    def __call__(self, func):
        for _ in range(self.budget):
            # Selection
            scores = [func(individual) for individual in self.population]
            probabilities = np.array(scores) / np.sum(scores)
            selected_indices = np.random.choice(self.population_size, size=self.population_size, p=probabilities)
            selected_population = [self.population[i] for i in selected_indices]

            # Crossover
            offspring_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(selected_population, 2)
                child = (parent1 + parent2) / 2
                offspring_population.append(child)

            # Mutation
            mutated_population = []
            for individual in offspring_population:
                if random.random() < 0.1:
                    mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
                    individual += mutation
                    individual = np.clip(individual, -5.0, 5.0)
                mutated_population.append(individual)

            # Replacement
            self.population = mutated_population
            self.population = np.array(self.population)
            self.population = np.sort(self.population, axis=0)

            # Update best individual
            for individual in self.population:
                if func(individual) < self.best_score:
                    self.best_individual = individual
                    self.best_score = func(individual)

            # Probabilistic adaptation
            if random.random() < 0.3:
                for i in range(self.population_size):
                    if random.random() < 0.5:
                        self.population[i] += random.uniform(-0.1, 0.1)
                        self.population[i] = np.clip(self.population[i], -5.0, 5.0)

        return self.best_individual, self.best_score

# Example usage
if __name__ == "__main__":
    budget = 100
    dim = 10
    func = lambda x: np.sum(x**2)
    algorithm = HybridEvolutionaryAlgorithm(budget, dim)
    best_individual, best_score = algorithm(func)
    print("Best Individual:", best_individual)
    print("Best Score:", best_score)