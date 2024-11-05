import numpy as np

class AdvancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.f = 0.8
        self.cr = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.reinit_threshold = budget // 8

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size

        dynamic_pop_size = self.pop_size  # Dynamic population size initialization
        while evals < self.budget:
            best_before = np.min(fitness)
            for i in range(dynamic_pop_size):  # Use dynamic population size
                self.f = 0.5 + 0.3 * np.sin(2 * np.pi * evals / self.budget + np.max(fitness))
                indices = np.random.choice(dynamic_pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                self.cr = 0.7 + 0.3 * (np.max(fitness) - fitness[i]) / (np.max(fitness) - np.min(fitness) + 1e-10)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

            if np.min(fitness) == best_before:
                dynamic_pop_size = max(4, dynamic_pop_size // 2)  # Adjust population size adaptively
            else:
                dynamic_pop_size = min(self.pop_size, dynamic_pop_size + 4)

            if evals % (self.budget // 10) == 0:  # Less frequent local search
                best_idx = np.argmin(fitness)
                local_search_vector = population[best_idx] + np.random.normal(0, 0.1, self.dim)  # Stochastic local search
                local_search_vector = np.clip(local_search_vector, self.lower_bound, self.upper_bound)
                local_fitness = func(local_search_vector)
                evals += 1
                if local_fitness < fitness[best_idx]:
                    population[best_idx] = local_search_vector
                    fitness[best_idx] = local_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]