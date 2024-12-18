import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30  # Increased initial population size
        self.CR = 0.7
        self.F = 0.8  # Adjusted mutation factor
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.initial_pop_size, self.dim))
        self.population_fitness = np.full(self.initial_pop_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        self.population_fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.initial_pop_size
        gen_counter = 0

        best_solution = None
        best_fitness = np.inf

        while self.evaluations < self.budget:
            for i in range(len(self.population)):
                idxs = [idx for idx in range(len(self.population)) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                dynamic_F = self.F * (0.75 + 0.25 * (1 - (self.evaluations / self.budget)))  # Adjusted dynamic scaling
                mutant = a + dynamic_F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)  # Simplified boundary handling

                adaptive_CR = self.CR + 0.1 * np.random.rand() * (1 - self.evaluations / self.budget)  # Adjusted CR
                crossover_mask = np.random.rand(self.dim) < adaptive_CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            current_best_idx = np.argmin(self.population_fitness)
            current_best_fitness = self.population_fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_solution = self.population[current_best_idx]
                best_fitness = current_best_fitness

            self.CR = 0.5 + 0.5 * np.random.rand()  # Adjusted random component for CR
            self.F = 0.5 + 0.3 * np.random.rand()  # Adjusted random component for F

            if self.evaluations % (self.budget // 10) == 0:  # Modified frequency of population resizing
                self.population = self.population[:max(5, int(len(self.population) * 0.9))]  # Finer adjustment
                self.population_fitness = self.population_fitness[:len(self.population)]

            gen_counter += 1

        return best_solution