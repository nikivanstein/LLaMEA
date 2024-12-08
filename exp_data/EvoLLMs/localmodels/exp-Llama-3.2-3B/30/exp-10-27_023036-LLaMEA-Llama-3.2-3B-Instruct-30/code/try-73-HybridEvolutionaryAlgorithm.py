import random
import numpy as np
import operator

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.probability_threshold = 0.3

    def __call__(self, func):
        # Initialize population
        population = [random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        for _ in range(self.budget):
            # Evaluate population
            scores = [func(x) for x in population]

            # Select best individuals
            selected_indices = np.argsort(scores)[-self.population_size:]
            selected_population = [population[i] for i in selected_indices]

            # Apply crossover and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(selected_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            # Replace worst individual with new individual
            worst_individual = min(new_population, key=lambda x: func(x))
            worst_index = population.index(worst_individual)
            population[worst_index] = new_population[np.argmin([func(x) for x in new_population])]

        # Return best individual
        return max(population, key=lambda x: func(x))

    def crossover(self, parent1, parent2):
        child = list(parent1)
        if random.random() < self.crossover_rate:
            idx = random.randint(0, self.dim - 1)
            child[idx] = parent2[idx]
        return child

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                individual[i] += random.uniform(-1.0, 1.0)
                individual[i] = max(-5.0, min(5.0, individual[i]))
        return individual

# Example usage
def func(x):
    return sum(x**2)

budget = 100
dim = 10
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
best_individual = algorithm(func)
print("Best individual:", best_individual)
print("Function value:", func(best_individual))