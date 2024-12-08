import numpy as np

class DynamicHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, self.budget // (2 * dim))
        self.harmony_memory_consideration_rate = 0.9
        self.adjustment_rate = 0.3
        self.bandwidth = 0.05
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def adaptive_mutation_factor(self):
        return 0.2 + 0.4 * np.random.rand()

    def harmony_search(self, func):
        for _ in range(self.budget // (3 * self.population_size)):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                new_solution = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.harmony_memory_consideration_rate:
                        new_solution[j] = self.population[np.random.randint(self.population_size), j]
                        if np.random.rand() < self.adjustment_rate:
                            new_solution[j] += self.bandwidth * np.random.uniform(-1, 1)
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_fitness = func(new_solution)
                self.evaluations += 1
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_solution
                if new_fitness < func(self.population[i]):
                    self.population[i] = new_solution

    def adaptive_differential_evolution(self, func):
        for _ in range(self.budget // (3 * self.population_size)):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutation_factor = self.adaptive_mutation_factor()
                mutant_vector = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.harmony_memory_consideration_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial_vector

    def stochastic_hill_climbing(self, func):
        step_size = 0.03
        for _ in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                direction = np.random.uniform(-1, 1, self.dim)
                candidate = self.population[i] + step_size * direction
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate
                if candidate_fitness < func(self.population[i]):
                    self.population[i] = candidate

    def __call__(self, func):
        self.harmony_search(func)
        self.adaptive_differential_evolution(func)
        self.stochastic_hill_climbing(func)
        return self.best_solution