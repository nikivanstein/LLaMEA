import numpy as np

class EnhancedAdaptiveLévyDifferentialEvolutionV8:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = max(5 * dim, 25)
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v)**(1 / beta)
        return step

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1

    def mutate(self, target_idx):
        indices = np.delete(np.arange(self.population_size), target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        best_idx = np.argmin(self.fitness)
        
        F = 0.6 + 0.3 * (1 - self.eval_count / self.budget) * np.random.rand()
        F *= np.random.choice([1.1, 0.9], p=[0.8, 0.2])
        
        if np.random.rand() < 0.4:
            mutant = self.population[best_idx] + F * (self.population[a] - self.population[b] + self.population[c] - self.population[best_idx])
        else:
            mutant = self.population[a] + F * (self.population[b] - self.population[c])
        
        return mutant

    def crossover(self, target, mutant):
        CR = 0.85 - 0.15 * (self.eval_count / self.budget)
        crossover = np.random.rand(self.dim) < CR
        return np.where(crossover, mutant, target)

    def select(self, target_idx, trial, func):
        trial_fitness = func(trial)
        self.eval_count += 1
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness

    def dynamic_population_resizing(self):
        if self.eval_count > self.budget * 0.5:
            self.population_size = max(int(self.initial_population_size * 0.6), 5 * self.dim)

    def __call__(self, func):
        self.evaluate_population(func)

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                if np.random.rand() < 0.15:
                    trial += 0.005 * self.levy_flight()

                self.select(i, trial, func)

            self.dynamic_population_resizing()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]