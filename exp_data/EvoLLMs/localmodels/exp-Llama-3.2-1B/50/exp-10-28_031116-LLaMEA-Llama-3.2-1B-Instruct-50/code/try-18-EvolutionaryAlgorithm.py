import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = [[random.uniform(-5.0, 5.0) for _ in range(dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        def fitness(individual):
            return np.sum(np.abs(func(individual)))

        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = []
            for _ in range(self.population_size):
                tournament_size = random.randint(1, self.population_size)
                tournament = random.choices(self.population, k=tournament_size, weight=[fitness(ind) for ind in self.population], key=lambda ind: random.random())
                parents.append(tournament[0])

            # Crossover and mutation
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[i+1]
                if random.random() < 0.5:
                    child = (parent1 + parent2) / 2
                else:
                    child = random.uniform(parent1, parent2)
                offspring.append(child)

            # Adapt population size and step size
            if len(offspring) < self.population_size / 2:
                self.population_size *= 2
            if self.population_size > 1000:
                self.population_size = 1000

            # Replace worst individuals with offspring
            worst_index = np.argmin(fitness(offspring))
            self.population[worst_index] = offspring[worst_index]

            # Update population size
            if len(self.population) > self.population_size:
                self.population_size = min(self.population_size, self.population_size * 0.8)

# Black box function
def func(x):
    return np.sum(np.abs(x))

# Example usage
ea = EvolutionaryAlgorithm(1000, 10)
best_func = np.inf
best_func_idx = -1
for _ in range(1000):
    best_func = min(best_func, ea(func))
    print(f"Best function: {best_func}, Best function index: {best_func_idx}")

# Update the algorithm with adaptive step size and population size
ea = EvolutionaryAlgorithm(1000, 10)
ea.budget = 500
ea.population_size = 200