import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.cr = 0.9  # Initial crossover probability
        self.f = 0.8   # Initial differential weight
        self.best_solution = None
        self.best_value = float('inf')
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.population_values = np.full(self.population_size, float('inf'))
        self.function_evals = 0
    
    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.population_values[i] == float('inf'):
                self.population_values[i] = func(self.population[i])
                self.function_evals += 1
                if self.population_values[i] < self.best_value:
                    self.best_value = self.population_values[i]
                    self.best_solution = np.copy(self.population[i])
    
    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant
    
    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial
    
    def select(self, target_idx, trial, func):
        trial_value = func(trial)
        self.function_evals += 1
        if trial_value < self.population_values[target_idx]:
            self.population[target_idx] = trial
            self.population_values[target_idx] = trial_value
            if trial_value < self.best_value:
                self.best_value = trial_value
                self.best_solution = np.copy(trial)
    
    def opposition_based_learning(self):
        for i in range(self.population_size):
            opposite_solution = self.lower_bound + self.upper_bound - self.population[i]
            opposite_solution = np.clip(opposite_solution, self.lower_bound, self.upper_bound)
            opposite_value = func(opposite_solution)
            self.function_evals += 1
            if opposite_value < self.population_values[i]:
                self.population[i] = opposite_solution
                self.population_values[i] = opposite_value
                if opposite_value < self.best_value:
                    self.best_value = opposite_value
                    self.best_solution = np.copy(opposite_solution)
    
    def adapt_params(self, generation):
        self.cr = 0.9 - 0.5 * (generation / (self.budget / self.population_size))
        self.f = 0.8 - 0.4 * (generation / (self.budget / self.population_size))
        self.f = max(self.f, 0.4)

    def __call__(self, func):
        generation = 0
        self.evaluate_population(func)
        self.opposition_based_learning()
        while self.function_evals < self.budget:
            self.adapt_params(generation)
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                self.select(i, trial, func)
            generation += 1
        return self.best_solution