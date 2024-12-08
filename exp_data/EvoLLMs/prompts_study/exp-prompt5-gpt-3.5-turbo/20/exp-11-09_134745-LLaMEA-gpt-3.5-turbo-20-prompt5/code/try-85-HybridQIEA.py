import numpy as np

class HybridQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def differential_evolution(self, best_individual):
        F = 0.5
        for idx in range(self.budget):
            r1, r2, r3 = np.random.choice(self.population, 3, replace=False)
            mutant = r1 + F * (r2 - r3)
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.5, mutant, self.population[idx])
            if func(trial) < func(self.population[idx]):
                self.population[idx] = trial

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            self.differential_evolution(best_individual)
        return best_individual