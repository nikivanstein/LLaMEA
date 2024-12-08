import numpy as np

class EnhancedImprovedADELSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.local_search_intensity = 5
        self.no_improvement_count = 0
        self.dynamic_pop_size = self.pop_size

    def __call__(self, func):
        evaluations = 0
        improvement = True

        for i in range(self.dynamic_pop_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if evaluations >= self.budget:
                return self.best_solution()

        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.dynamic_pop_size):
                indices = list(range(self.dynamic_pop_size))
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
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness
                    improvement = True
                else:
                    new_population[i] = self.population[i]
                
                if evaluations >= self.budget:
                    return self.best_solution()

            self.population = np.vstack((new_population, self.create_random_solutions()))
            self.dynamic_pop_size = len(self.population)
            
            if improvement:
                self.local_search_intensity = max(1, self.local_search_intensity - 1)
                improvement = False
                self.no_improvement_count = 0
            else:
                self.local_search_intensity = min(10, self.local_search_intensity + 1)
                self.no_improvement_count += 1

            if evaluations + self.local_search_intensity <= self.budget:
                self.hybrid_local_search(func)
                evaluations += self.local_search_intensity

            if self.no_improvement_count > 5:
                random_idx = np.random.choice(self.dynamic_pop_size, 1)
                self.population[random_idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                self.fitness[random_idx] = func(self.population[random_idx])
                evaluations += 1
                self.no_improvement_count = 0

        return self.best_solution()

    def create_random_solutions(self):
        num_new_solutions = max(1, self.dynamic_pop_size // 10)
        new_solutions = np.random.uniform(self.lower_bound, self.upper_bound, (num_new_solutions, self.dim))
        return new_solutions

    def hybrid_local_search(self, func):
        best_idx = np.argmin(self.fitness)
        best_candidate = self.population[best_idx]
        perturbation = np.random.normal(0, 0.1, (self.local_search_intensity, self.dim))
        local_candidates = best_candidate + perturbation
        local_candidates = np.clip(local_candidates, self.lower_bound, self.upper_bound)
        local_fitnesses = np.apply_along_axis(func, 1, local_candidates)
        min_idx = np.argmin(local_fitnesses)
        if local_fitnesses[min_idx] < self.fitness[best_idx]:
            self.population[best_idx] = local_candidates[min_idx]
            self.fitness[best_idx] = local_fitnesses[min_idx]

    def best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]