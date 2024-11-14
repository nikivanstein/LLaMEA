import numpy as np

class AdaptiveMemoryDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.best_solution = None
        self.best_value = float('inf')
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.population_values = np.full(self.population_size, float('inf'))
        self.function_evals = 0
        self.memory_factor = 0.5
        self.memory = []

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.population_values[i] == float('inf'):
                self.population_values[i] = func(self.population[i])
                self.function_evals += 1
                if self.population_values[i] < self.best_value:
                    self.best_value = self.population_values[i]
                    self.best_solution = np.copy(self.population[i])
                    self.memory.append(self.best_solution)

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
                self.memory.append(self.best_solution)

    def adapt_params(self, generation):
        self.cr = 0.9 - 0.5 * (generation / (self.budget / self.initial_population_size))
        self.f = 0.8 - 0.4 * (generation / (self.budget / self.initial_population_size))

    def levy_flight(self, step_factor=0.01):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / 3)
        return step_factor * step

    def enhance_exploration(self):
        if len(self.memory) > 0 and np.random.rand() < self.memory_factor:
            mem_idx = np.random.randint(0, len(self.memory))
            direction = self.levy_flight()
            new_sol = self.memory[mem_idx] + direction
            new_sol = np.clip(new_sol, self.lower_bound, self.upper_bound)
            return new_sol
        else:
            return None

    def __call__(self, func):
        generation = 0
        self.evaluate_population(func)
        while self.function_evals < self.budget:
            self.adapt_params(generation)
            self.resize_population()
            for i in range(self.population_size):
                if self.function_evals >= self.budget:
                    break
                enhanced_solution = self.enhance_exploration()
                if enhanced_solution is not None:
                    trial = self.crossover(self.population[i], enhanced_solution)
                else:
                    mutant = self.mutate(i)
                    trial = self.crossover(self.population[i], mutant)
                self.select(i, trial, func)
            generation += 1
        return self.best_solution