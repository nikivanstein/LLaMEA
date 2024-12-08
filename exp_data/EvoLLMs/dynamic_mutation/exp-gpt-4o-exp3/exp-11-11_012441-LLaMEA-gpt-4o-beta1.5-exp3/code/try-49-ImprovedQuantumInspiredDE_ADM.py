import numpy as np

class ImprovedQuantumInspiredDE_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f = 0.8
        self.dynamic_f_range = [0.5, 0.9]
        self.best_solution = None
        self.best_value = float('inf')
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.population_values = np.full(self.population_size, float('inf'))
        self.function_evals = 0
        self.memory = {'cr': [], 'f': [], 'success': []}
        self.global_best = np.copy(self.population[np.argmin(self.population_values)])

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
        dynamic_f = np.random.uniform(self.dynamic_f_range[0], self.dynamic_f_range[1])
        mutant = self.population[a] + dynamic_f * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        adapt_cr = np.random.normal(self.cr, 0.1)
        crossover_mask = np.random.rand(self.dim) < adapt_cr
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial, func):
        trial_value = func(trial)
        self.function_evals += 1
        improvement = trial_value < self.population_values[target_idx]
        if improvement:
            self.population[target_idx] = trial
            self.population_values[target_idx] = trial_value
            if trial_value < self.best_value:
                self.best_value = trial_value
                self.best_solution = np.copy(trial)
        return improvement

    def adapt_params(self):
        if len(self.memory['success']) > 0:
            success_rate = np.mean(self.memory['success'][-5:])
            if success_rate > 0.2:
                self.cr = np.clip(self.cr + np.random.normal(0, 0.1 * (success_rate - 0.2)), 0.3, 0.9)
            self.f = np.clip(self.f + np.random.normal(0, 0.1 * success_rate), 0.4, 0.9)

    def quantum_inspired_update(self, idx):
        prob = np.random.rand()
        if prob < 0.5:
            self.population[idx] = 0.5 * (self.population[idx] + self.global_best)
        else:
            self.population[idx] = self.population[idx] + np.random.normal(0, 0.2, self.dim) * (self.global_best - self.population[idx])
        self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.evaluate_population(func)
        while self.function_evals < self.budget:
            self.adapt_params()
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                self.quantum_inspired_update(i)
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                success = self.select(i, trial, func)
                self.memory['success'].append(success)
                if success:
                    self.memory['cr'].append(self.cr)
                    self.memory['f'].append(self.f)
            self.global_best = self.population[np.argmin(self.population_values)]
        return self.best_solution