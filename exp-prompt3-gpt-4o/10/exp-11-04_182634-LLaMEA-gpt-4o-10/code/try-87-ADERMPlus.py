import numpy as np

class ADERMPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0
        self.success_history = []

    def _mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        novelty_factor = np.std(self.population, axis=0).mean()  # Novelty-driven factor
        adaptive_F = self.F * (1 + novelty_factor * np.random.uniform(-0.1, 0.1))
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        success_rate = np.mean(self.success_history) if self.success_history else 0.5
        adaptive_CR = self.CR * success_rate  # Adaptive based on historical success
        crossover = np.random.rand(self.dim) < adaptive_CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def __call__(self, func):
        dim_scale_factor = np.random.uniform(0.8, 1.2)
        scaled_dim = max(1, int(self.dim * dim_scale_factor))
        self.fitness = np.array([func(ind) for ind in self.population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            best_idx = np.argmin(self.fitness)
            best_individual = self.population[best_idx]

            for i in range(self.population_size):
                if i != best_idx:
                    mutant = self._mutate(i)
                    trial = self._crossover(self.population[i], mutant)
                    trial_fitness = func(trial)
                    self.eval_count += 1
                    success = trial_fitness < self.fitness[i]
                    self.success_history.append(success)

                    if success:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness

                    if self.eval_count >= self.budget:
                        break

            if self.eval_count > self.budget / 2:
                self.population_size = max(int(0.5 * self.initial_population_size), 4)
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx][:scaled_dim]