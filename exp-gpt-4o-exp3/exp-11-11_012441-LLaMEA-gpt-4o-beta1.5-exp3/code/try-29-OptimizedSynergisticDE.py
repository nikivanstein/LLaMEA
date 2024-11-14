import numpy as np

class OptimizedSynergisticDE:
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
        self.memory = {'cr': [], 'f': []}
        self.niches = np.random.uniform(self.lower_bound, self.upper_bound, (5, dim))
        self.niche_radius = 1.0

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.population_values[i] == float('inf'):
                self.population_values[i] = func(self.population[i])
                self.function_evals += 1
                if self.population_values[i] < self.best_value:
                    self.best_value = self.population_values[i]
                    self.best_solution = np.copy(self.population[i])

    def mutate(self, idx, best_idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b = np.random.choice(indices, 2, replace=False)
        mutant = self.population[idx] + self.f * (self.population[best_idx] - self.population[idx]) + self.f * (self.population[a] - self.population[b])
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
            self.population[target_idx] = trial
            self.population_values[target_idx] = trial_value
            if trial_value < self.best_value:
                self.best_value = trial_value
                self.best_solution = np.copy(trial)
        return improvement

    def adapt_params(self, generation):
        if len(self.memory['cr']) > 5:
            self.cr = np.mean(self.memory['cr'][-5:])
        if len(self.memory['f']) > 5:
            self.f = np.mean(self.memory['f'][-5:])
        oscillation_factor = (1 + np.cos(2 * np.pi * generation / 50)) / 2
        self.f = 0.6 + 0.2 * oscillation_factor

    def dynamic_niche_preservation(self):
        for idx in range(self.population_size):
            distances = np.linalg.norm(self.niches - self.population[idx], axis=1)
            if np.any(distances < self.niche_radius):
                self.population[idx] += np.random.uniform(-0.1, 0.1, self.dim) * (self.upper_bound - self.lower_bound)

    def fitness_based_scaling(self):
        sorted_indices = np.argsort(self.population_values)
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]
        fitness_range = self.population_values[worst_idx] - self.population_values[best_idx]
        if fitness_range != 0:
            scaled_f = (self.population_values[worst_idx] - self.population_values) / fitness_range
            self.f = 0.5 + 0.5 * scaled_f

    def __call__(self, func):
        generation = 0
        self.evaluate_population(func)
        while self.function_evals < self.budget:
            self.adapt_params(generation)
            self.dynamic_niche_preservation()
            self.fitness_based_scaling()
            best_idx = np.argmin(self.population_values)
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                mutant = self.mutate(i, best_idx)
                trial = self.crossover(self.population[i], mutant)
                success = self.select(i, trial, func)
                if success:
                    self.memory['cr'].append(self.cr)
                    self.memory['f'].append(self.f)
            generation += 1
        return self.best_solution