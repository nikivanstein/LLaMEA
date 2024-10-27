import numpy as np
import random

class HybridHarmonySearchDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.harmony_size = 10
        self.harmony = self.initialize_harmony()
        self.best_solution = self.initialize_best_solution()

    def initialize_harmony(self):
        return np.random.uniform(-5.0, 5.0, (self.harmony_size, self.dim))

    def initialize_best_solution(self):
        return np.random.uniform(-5.0, 5.0, (1, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.harmony)
        self.harmony = self.select_harmonies(fitnesses)
        self.harmony = self.crossover(self.harmony)
        self.harmony = self.mutate(self.harmony)
        self.update_best_solution(fitnesses)

    def select_harmonies(self, fitnesses):
        fitnesses = np.array(fitnesses)
        harmonies = np.array([self.harmony[np.argsort(fitnesses)[:int(self.harmony_size/2)]]])
        return harmonies

    def crossover(self, harmonies):
        offspring = np.zeros((self.harmony_size, self.dim))
        for i in range(self.harmony_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(harmonies)
                parent2 = random.choice(harmonies)
                offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, harmonies):
        mutated_harmony = np.copy(harmonies)
        for i in range(self.harmony_size):
            if random.random() < self.mutation_probability:
                mutated_harmony[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_harmony

    def update_best_solution(self, fitnesses):
        fitnesses = np.array(fitnesses)
        self.best_solution = np.argmin(fitnesses)

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return self.best_solution

# Example usage
def func(x):
    return np.sum(x**2)

hhsde = HybridHarmonySearchDE(budget=100, dim=10)
optimal_solution = hhsde(func)
print(optimal_solution)