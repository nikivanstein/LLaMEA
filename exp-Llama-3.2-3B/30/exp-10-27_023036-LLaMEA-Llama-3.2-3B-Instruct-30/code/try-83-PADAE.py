import numpy as np
import random

class PADAE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness = []

    def __call__(self, func):
        # Initialize population with random candidates
        for _ in range(self.budget):
            candidate = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(candidate)
            self.fitness.append(func(candidate))

        # Dynamic evolution
        for _ in range(self.budget):
            # Select parents based on fitness
            parents = np.array(self.population)[np.argsort(self.fitness)]
            parents = parents[:int(self.budget/2)]

            # Crossover and mutation
            offspring = []
            while len(offspring) < self.budget - len(parents):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2 + np.random.uniform(-0.5, 0.5, self.dim)
                offspring.append(child)

            # Replace worst individual
            worst_idx = np.argmin(self.fitness)
            self.population[worst_idx] = offspring[worst_idx]

            # Update fitness
            self.fitness[worst_idx] = func(offspring[worst_idx])

            # Adaptive probability
            prob = 1 - (len(offspring) / self.budget)
            if random.random() < prob:
                # Randomly replace an offspring
                offspring_idx = random.randint(0, len(offspring) - 1)
                self.population[offspring_idx] = np.random.uniform(-5.0, 5.0, self.dim)
                self.fitness[offspring_idx] = func(self.population[offspring_idx])

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
pada = PADAE(budget, dim)
best_solution = None
best_score = -np.inf
for _ in range(100):
    solution = np.random.uniform(-5.0, 5.0, dim)
    score = func(solution)
    if score < best_score:
        best_solution = solution
        best_score = score

    # Refine the best solution
    if best_solution is not None:
        pada(solution)
        padad.fitness = [score for score in padad.fitness if score!= score]  # Remove NaN values
        padad.population = [solution] * len(padad.population)
        padad.fitness = [score for score in padad.fitness if score!= score]  # Remove NaN values
        best_solution = None
        best_score = -np.inf

print("Best solution:", best_solution)
print("Best score:", best_score)