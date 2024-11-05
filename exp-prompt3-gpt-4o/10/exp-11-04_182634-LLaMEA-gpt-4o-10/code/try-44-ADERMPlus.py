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

    def _levy_flight(self, idx):
        # Levy flight calculation
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / beta)
        return step

    def _mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * (1 + np.random.uniform(-0.1, 0.1))  # Adaptive scaling
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        
        # Levy flight enhancement
        if np.random.rand() < 0.3:  # 30% chance to apply Levy flight
            mutant += 0.01 * self._levy_flight(idx)
        
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        adaptive_CR = self.CR * (1 + np.random.uniform(-0.05, 0.05))  # Adaptive crossover rate
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