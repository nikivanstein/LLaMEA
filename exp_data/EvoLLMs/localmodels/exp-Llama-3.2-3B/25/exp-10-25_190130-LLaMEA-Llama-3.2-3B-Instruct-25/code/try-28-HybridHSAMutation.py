import numpy as np
import random
import math

class HybridHSAMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.mutation_rate = 0.1
        self.harmony_size = 50
        self.crossover_rate = 0.8
        self.probability = 0.25
        self.harmony = self.initialize_harmony()
        self.best_solution = None

    def initialize_harmony(self):
        harmony = []
        for _ in range(self.population_size):
            solution = [random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
            harmony.append(solution)
        return harmony

    def fitness(self, solution):
        return func(solution)

    def evaluate(self):
        fitness_values = [self.fitness(solution) for solution in self.harmony]
        best_index = np.argmax(fitness_values)
        self.harmony[best_index] = self.adapt_best_solution(best_index)
        return fitness_values

    def adapt_best_solution(self, index):
        solution = self.harmony[index]
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                new_value = solution[i] + random.uniform(-1, 1)
                if random.random() < self.probability:
                    new_value = max(self.lower_bound, min(new_value, self.upper_bound))
                solution[i] = new_value
        return solution

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.dim):
            if random.random() < self.crossover_rate:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def __call__(self, func):
        for _ in range(self.budget):
            if self.best_solution is None or self.fitness(self.best_solution) < self.fitness(self.harmony[0]):
                self.best_solution = self.harmony[0]
            self.harmony = self.evaluate()
            new_harmony = []
            for i in range(self.population_size):
                parent1 = random.choice(self.harmony)
                parent2 = random.choice(self.harmony)
                child = self.crossover(parent1, parent2)
                new_harmony.append(child)
            self.harmony = new_harmony
            print(f'Iteration {_+1}, Best Solution: {self.best_solution}, Fitness: {self.fitness(self.best_solution)}')
        return self.best_solution

def func(solution):
    return sum([s**2 for s in solution])

# Example usage:
budget = 100
dim = 10
hsa = HybridHSAMutation(budget, dim)
best_solution = hsa(func)
print(f'Best Solution: {best_solution}, Fitness: {func(best_solution)}')