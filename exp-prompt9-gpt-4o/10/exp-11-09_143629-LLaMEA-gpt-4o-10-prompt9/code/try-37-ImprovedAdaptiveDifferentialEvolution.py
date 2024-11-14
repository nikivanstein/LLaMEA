import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_pop_size = pop_size if pop_size is not None else 8 * dim
        self.max_pop_size = int(self.initial_pop_size * 1.2)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.eval_count = 0
        self.global_best = None

    def evaluate_population(self, func):
        for i in range(self.population.shape[0]):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1
                if self.global_best is None or self.fitness[i] < self.fitness[self.global_best]:
                    self.global_best = i

    def select_parents(self):
        indices = np.random.choice(self.population.shape[0], 3, replace=False)
        return self.population[indices]

    def mutate(self, a, b, c, F=0.5 + np.random.rand() * 0.5):  # Adjusted F factor for enhanced exploration
        mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, CR=0.9):  # Slightly increased CR for enhanced diversity
        crossover_mask = np.random.rand(self.dim) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def adaptive_random_search(self):
        perturbation = np.random.uniform(-0.3, 0.3, self.dim)  # Adjusted perturbation range
        candidate = np.clip(self.population[self.global_best] + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def competitive_selection(self, trial, trial_fitness, target_idx):
        if trial_fitness < self.fitness[target_idx]:
            return trial, trial_fitness
        else:
            return self.population[target_idx], self.fitness[target_idx]

    def dynamic_population_size(self):
        new_size = min(self.max_pop_size, self.population.shape[0] + 1)
        if new_size > self.population.shape[0]:
            new_individual = np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dim))
            self.population = np.vstack([self.population, new_individual])
            self.fitness = np.append(self.fitness, np.inf)

    def optimize(self, func):
        self.evaluate_population(func)
        while self.eval_count < self.budget:
            self.dynamic_population_size()
            for i in range(self.population.shape[0]):
                if np.random.rand() < 0.25:  # Adjusted chance to use adaptive random search
                    trial = self.adaptive_random_search()
                else:
                    target = self.population[i]
                    a, b, c = self.select_parents()
                    mutant = self.mutate(a, b, c)
                    trial = self.crossover(target, mutant)
                
                trial_fitness = func(trial)
                self.eval_count += 1

                self.population[i], self.fitness[i] = self.competitive_selection(trial, trial_fitness, i)

                if self.fitness[i] < self.fitness[self.global_best]:
                    self.global_best = i

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def __call__(self, func):
        best_solution, best_fitness = self.optimize(func)
        return best_solution, best_fitness