import numpy as np
from sklearn.cluster import KMeans

class ADELSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Initial crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.local_search_intensity = 5

    def __call__(self, func):
        evaluations = 0
        improvement = True

        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if evaluations >= self.budget:
                return self.best_solution()

        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.pop_size):
                if evaluations < self.budget // 2:
                    self.CR = 0.9
                else:
                    self.CR = 0.5 + 0.4 * np.random.rand()
                
                # Select indices for mutation
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Perform mutation and crossover
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
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
                    improvement = True
                else:
                    new_population[i] = self.population[i]
                
                if evaluations >= self.budget:
                    return self.best_solution()

            self.population = new_population

            # Dynamic Adaptation of Local Search Intensity
            if improvement:
                self.local_search_intensity = max(1, self.local_search_intensity - 1)
                improvement = False
            else:
                self.local_search_intensity = min(10, self.local_search_intensity + 1)

            # Apply local search to cluster centers periodically if budget allows
            if evaluations + self.local_search_intensity <= self.budget:
                kmeans = KMeans(n_clusters=5).fit(self.population)
                centers = kmeans.cluster_centers_
                for center in centers:
                    self.local_search(center, func)
                    evaluations += self.local_search_intensity

        return self.best_solution()

    def local_search(self, candidate, func):
        for _ in range(self.local_search_intensity):
            perturbation = np.random.normal(0, 0.1, self.dim)
            local_candidate = candidate + perturbation
            local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
            local_fitness = func(local_candidate)
            # Update the closest individual in population
            idx = np.argmin(np.linalg.norm(self.population - local_candidate, axis=1))
            if local_fitness < self.fitness[idx]:
                self.population[idx] = local_candidate
                self.fitness[idx] = local_fitness

    def best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]