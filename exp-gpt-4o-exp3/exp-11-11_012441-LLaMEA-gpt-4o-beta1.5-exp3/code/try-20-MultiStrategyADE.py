import numpy as np

class MultiStrategyADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.cr = 0.9
        self.f = 0.8
        self.best_solution = None
        self.best_value = float('inf')
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.population_values = np.full(self.population_size, float('inf'))
        self.function_evals = 0
        self.archive = []
        self.history = {'mutation': [], 'crossover': []}
        self.success_rates = {'mutation': 0, 'crossover': 0}

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
        improvement = trial_value < self.population_values[target_idx]
        if improvement:
            self.archive.append(self.population[target_idx])
            self.population[target_idx] = trial
            self.population_values[target_idx] = trial_value
            if trial_value < self.best_value:
                self.best_value = trial_value
                self.best_solution = np.copy(trial)
        return improvement

    def adapt_params(self, generation):
        oscillation_factor = (1 + np.cos(2 * np.pi * generation / 50)) / 2
        self.cr = 0.5 + 0.4 * oscillation_factor
        self.f = 0.6 + 0.2 * oscillation_factor

    def dynamic_operator_selection(self):
        if len(self.history['mutation']) > 10:
            recent_mutation_success = np.mean(self.history['mutation'][-10:])
            recent_crossover_success = np.mean(self.history['crossover'][-10:])
            if recent_mutation_success < recent_crossover_success:
                factor_increase = 0.05
                self.f = min(self.f + factor_increase, 1.0)
                self.cr = max(self.cr - factor_increase, 0.0)
            else:
                factor_increase = 0.05
                self.cr = min(self.cr + factor_increase, 1.0)
                self.f = max(self.f - factor_increase, 0.0)

    def resize_population(self):
        stochastic_factor = np.random.uniform(0.6, 0.8)
        if self.function_evals > 0.5 * self.budget:
            new_size = max(self.dim, int(self.population_size * stochastic_factor))
            if new_size < self.population_size:
                indices = np.argsort(self.population_values)[:new_size]
                self.population = self.population[indices]
                self.population_values = self.population_values[indices]
                self.population_size = new_size
                self.archive = []

    def contextual_learning(self):
        if len(self.archive) > 0:
            best_archive = min(self.archive, key=lambda x: func(x))
            if func(best_archive) < self.best_value:
                self.best_solution = np.copy(best_archive)
                self.best_value = func(best_archive)
                self.archive = []

    def __call__(self, func):
        generation = 0
        self.evaluate_population(func)
        while self.function_evals < self.budget:
            self.adapt_params(generation)
            self.dynamic_operator_selection()
            self.resize_population()
            self.contextual_learning()
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                success = self.select(i, trial, func)
                self.history['mutation'].append(success)
                self.history['crossover'].append(success)
            generation += 1
        return self.best_solution