import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(10 * dim, max(20, budget // 50))  # Adaptive pop size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0
        self.learning_rate = 0.1  # Learning rate for mutation factor adjustment
        
    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def select_parents(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        return np.random.choice(indices, 3, replace=False)

    def mutate(self, idx, r1, r2, r3):
        mutation_factor = self.mutation_factor * (1 - self.evaluations / self.budget)
        mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def crossover(self, idx, mutant):
        trial = np.copy(self.population[idx])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.crossover_rate or j == j_rand:
                trial[j] = mutant[j]
        return trial
    
    def optimize(self, func):
        self.evaluate_population(func)
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2, r3 = self.select_parents(i)
                mutant = self.mutate(i, r1, r2, r3)
                trial = self.crossover(i, mutant)

                if self.evaluations < self.budget:
                    trial_fitness = func(trial)
                    self.evaluations += 1

                    if trial_fitness < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness
                        self.mutation_factor += self.learning_rate * (trial_fitness - self.fitness[i])

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
    
    def __call__(self, func):
        return self.optimize(func)