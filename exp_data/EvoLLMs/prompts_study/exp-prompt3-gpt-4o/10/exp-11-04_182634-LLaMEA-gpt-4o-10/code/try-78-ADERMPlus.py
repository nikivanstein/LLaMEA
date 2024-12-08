import numpy as np

class ADERMPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0
        self.memory_F = np.zeros(self.population_size)  # Track historical F values
        self.memory_CR = np.zeros(self.population_size)  # Track historical CR values

    def _mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        fitness_factor = 1.0 - (self.fitness[idx] / (np.max(self.fitness) + 1e-9))
        adaptive_F = self.F * (1 + fitness_factor * np.random.uniform(-0.1, 0.1))  # Fitness-based adaptive scaling
        self.memory_F[idx] = adaptive_F  # Update historical memory
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        crossover_range = np.linspace(0.85, 0.95, num=10)
        adaptive_CR = crossover_range[int((self.budget - self.eval_count) / self.budget * 9)]  # Dynamic crossover tuning
        self.memory_CR[:] = adaptive_CR  # Update historical memory for CR
        crossover = np.random.rand(self.dim) < adaptive_CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def __call__(self, func):
        dim_scale_factor = np.random.uniform(0.8, 1.2)  # Adaptive dimension scaling
        scaled_dim = max(1, int(self.dim * dim_scale_factor))
        self.fitness = np.array([func(ind) for ind in self.population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            best_idx = np.argmin(self.fitness)  # Elitism: Preserve the best individual
            best_individual = self.population[best_idx]

            for i in range(self.population_size):
                if i != best_idx:  # Skip mutation for the best individual
                    mutant = self._mutate(i)
                    trial = self._crossover(self.population[i], mutant)
                    trial_fitness = func(trial)
                    self.eval_count += 1

                    if trial_fitness < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness

                    if self.eval_count >= self.budget:
                        break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx][:scaled_dim]