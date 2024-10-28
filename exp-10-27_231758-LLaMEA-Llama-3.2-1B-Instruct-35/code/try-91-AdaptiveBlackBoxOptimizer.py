import numpy as np
import random
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_black_box(self, func, bounds, initial_guess, mutation_rate, selection_rate, elite_size):
        population_size = int(self.budget / (mutation_rate + selection_rate))
        population = [initial_guess]
        elite = [population[0]]
        for _ in range(population_size - len(elite)):
            for _ in range(population_size - len(elite)):
                parent1, parent2 = random.sample(population, 2)
                child = self.mutate(parent1, parent2, mutation_rate)
                population.append(child)
                if len(elite) < elite_size:
                    elite.append(child)

        while len(elite) < elite_size:
            parent1, parent2 = random.sample(elite, 2)
            child = self.mutate(parent1, parent2, mutation_rate)
            population.append(child)

        best_func = None
        best_score = float('inf')
        for func in population:
            score = func(self.func_values)
            if score < best_score:
                best_func = func
                best_score = score
        return best_func

    def mutate(self, parent1, parent2, mutation_rate):
        idx = random.randint(0, self.dim - 1)
        new_value = parent1[idx] + random.uniform(-1, 1)
        return new_value

# Description: Adaptive Black Box Optimizer
# Code: 