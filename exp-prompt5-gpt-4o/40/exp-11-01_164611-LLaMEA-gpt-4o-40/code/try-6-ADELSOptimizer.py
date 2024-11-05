import numpy as np

class ADELSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.local_search_intensity = 5

    def __call__(self, func):
        evaluations = 0

        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if evaluations >= self.budget:
                return self.best_solution()

        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.pop_size):
                # Select indices for mutation
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Perform mutation with adaptive factor
                adaptive_F = self.F + (0.9 - self.F) * (self.fitness[i] - min(self.fitness)) / (max(self.fitness) - min(self.fitness))
                mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness
                else:
                    new_population[i] = self.population[i]
                
                if evaluations >= self.budget:
                    return self.best_solution()

            self.population = new_population

            # Dynamic local search intensity based on budget
            dynamic_intensity = max(1, int(self.local_search_intensity * (1 - evaluations / self.budget)))
            if evaluations + dynamic_intensity <= self.budget:
                best_idx = np.argmin(self.fitness)
                self.local_search(best_idx, func, dynamic_intensity)
                evaluations += dynamic_intensity

        return self.best_solution()

    def local_search(self, index, func, intensity):
        candidate = self.population[index]
        for _ in range(intensity):
            perturbation = np.random.normal(0, 0.1, self.dim)
            local_candidate = candidate + perturbation
            local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
            local_fitness = func(local_candidate)
            if local_fitness < self.fitness[index]:
                self.population[index] = local_candidate
                self.fitness[index] = local_fitness

    def best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]