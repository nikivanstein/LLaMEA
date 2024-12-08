import numpy as np

class DynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.F = 0.5
        self.CR = 0.9
        self.min_mutation = 0.5
        self.max_mutation = 2.0

    def __call__(self, func):
        def mutate(x, u, v):
            mutant = x + self.F * (u - v)
            return np.clip(mutant, -5.0, 5.0)

        def dynamic_mutation_scale(iteration, max_iterations):
            return self.min_mutation + (self.max_mutation - self.min_mutation) * (1 - iteration / max_iterations)

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        for _ in range(self.budget // self.population_size):
            for _ in range(self.population_size):
                target = np.random.choice(self.population)
                r1, r2, r3 = np.random.choice(self.population, 3, replace=False)
                mutation_scale = dynamic_mutation_scale(_, self.budget // self.population_size)
                mutant = mutate(target, r1, r2)
                trial = np.where(np.random.uniform(size=self.dim) < self.CR, mutant, target)
                fitness = func(trial)
                if fitness < best_fitness:
                    best_solution, best_fitness = trial, fitness
            self.population.append(best_solution)
        return best_solution