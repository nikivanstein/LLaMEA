import numpy as np

class ImprovedADELSOptimizer:
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
        self.no_improvement_count = 0
        self.elite_fraction = 0.1  # Fraction of elitism

    def __call__(self, func):
        evaluations = 0
        improvement = True

        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if evaluations >= self.budget:
                return self.best_solution()

        while evaluations < self.budget:
            elite_count = int(self.pop_size * self.elite_fraction)
            elite_indices = np.argpartition(self.fitness, elite_count)[:elite_count]
            new_population = np.copy(self.population[elite_indices])

            for i in range(self.pop_size - elite_count):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                F_adaptive = 0.1 + 0.9 * np.random.rand()
                mutant = self.population[a] + F_adaptive * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]
                
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    new_population = np.vstack((new_population, trial))
                    self.fitness[i] = trial_fitness
                    improvement = True
                else:
                    new_population = np.vstack((new_population, self.population[i]))
                
                if evaluations >= self.budget:
                    return self.best_solution()

            self.population = new_population[:self.pop_size]

            if improvement:
                self.local_search_intensity = max(1, self.local_search_intensity - 1)
                improvement = False
                self.no_improvement_count = 0
            else:
                self.local_search_intensity = min(10, self.local_search_intensity + 1)
                self.no_improvement_count += 1

            if evaluations + self.local_search_intensity <= self.budget:
                best_idx = np.argmin(self.fitness)
                self.local_search(best_idx, func)
                evaluations += self.local_search_intensity

            if self.no_improvement_count > 5:
                random_idx = np.random.choice(self.pop_size, 1)
                self.population[random_idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                self.fitness[random_idx] = func(self.population[random_idx])
                evaluations += 1
                self.no_improvement_count = 0

        return self.best_solution()

    def local_search(self, index, func):
        candidate = self.population[index]
        for _ in range(self.local_search_intensity):
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