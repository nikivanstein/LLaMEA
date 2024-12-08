import random
import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitnesses = []

    def __call__(self, func):
        for _ in range(self.budget):
            func_value = func()
            if len(self.population) >= self.budget:
                fitnesses = [self.fitnesses[i] for i in range(len(self.population))]
                best_idx = np.argmin(fitnesses)
                best_func = self.population[best_idx]
                best_func_value = best_func_value = func_value
                for i in range(self.dim):
                    mutation_rate = random.random()
                    if mutation_rate < 0.1:
                        mutation_idx = random.randint(0, len(self.population) - 1)
                        self.population[mutation_idx].fitnesses[i] += 1
                self.population[best_idx].fitnesses[i] += 1
                self.fitnesses.append(best_func_value)
                return best_func
            else:
                new_func = func()
                self.population.append(new_func)
                self.fitnesses.append(new_func_value)

    def select_solution(self):
        if len(self.population) < self.budget:
            return random.choice(self.population)
        else:
            return random.choices(self.population, weights=self.fitnesses, k=1)[0]

    def mutate(self, func):
        if random.random() < 0.1:
            func_value = func()
            mutation_idx = random.randint(0, len(self.population) - 1)
            self.population[mutation_idx].fitnesses[-1] += 1
            self.population[mutation_idx].fitnesses[0] += 1
            self.population[mutation_idx].fitnesses[-2] += 1
            self.population[mutation_idx].fitnesses[1] += 1
        return func

# Description: Evolutionary Algorithm with Adaptive Learning Rate and Mutation
# Code: 