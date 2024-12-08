import numpy as np

class HybridQuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability

    def quantum_superposition(self, sol):
        theta = np.random.uniform(0, 2 * np.pi, self.dim)
        return sol + np.cos(theta) * (self.upper_bound - self.lower_bound) / 2

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.cr
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        return np.where(crossover, mutant, target)

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                trial = self.quantum_superposition(trial)  # Apply quantum superposition
                
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial
                    self.best_fitness = trial_fitness
                
                if trial_fitness < func(target):
                    self.population[i] = trial
                
                if eval_count >= self.budget:
                    break

        return self.best_solution, self.best_fitness