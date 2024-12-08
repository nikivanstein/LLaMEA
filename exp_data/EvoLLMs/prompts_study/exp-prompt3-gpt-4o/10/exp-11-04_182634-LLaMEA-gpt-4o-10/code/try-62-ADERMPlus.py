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

    def _mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        oscillate_factor = np.sin(0.1 * self.eval_count)  # Oscillating factor
        adaptive_F = self.F * (1 + oscillate_factor)  # Adaptive scaling
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        oscillate_factor = np.sin(0.1 * self.eval_count)  # Oscillating factor
        adaptive_CR = self.CR * (1 + oscillate_factor)  # Adaptive crossover rate
        crossover = np.random.rand(self.dim) < adaptive_CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def _local_search(self, individual):
        perturbation = np.random.normal(0, 0.1, self.dim)
        candidate = np.clip(individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def __call__(self, func):
        dim_scale_factor = np.random.uniform(0.8, 1.2)  # Adaptive dimension scaling
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

                    if trial_fitness < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness

                    # Local search for refined exploitation
                    if i % 5 == 0 and trial_fitness < self.fitness[i]:
                        local_candidate = self._local_search(self.population[i])
                        local_fitness = func(local_candidate)
                        self.eval_count += 1
                        if local_fitness < self.fitness[i]:
                            self.population[i] = local_candidate
                            self.fitness[i] = local_fitness

                    if self.eval_count >= self.budget:
                        break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx][:scaled_dim]