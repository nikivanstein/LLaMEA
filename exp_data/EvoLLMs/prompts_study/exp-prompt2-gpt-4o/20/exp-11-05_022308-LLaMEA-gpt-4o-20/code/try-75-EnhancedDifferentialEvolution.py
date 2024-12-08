import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.f = 0.8
        self.cr = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.reinit_threshold = budget // 6  # Adjusted reinitialization threshold

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        eval_step = self.budget // (4 * self.pop_size)  # Adjusted dynamic eval step
        no_improvement_count = 0
        swarm_best = np.min(fitness)

        while evals < self.budget:
            best_before = np.min(fitness)
            for i in range(self.pop_size):
                self.f = 0.5 + 0.5 * np.random.rand() * np.sin(np.pi * evals / self.budget)  # Modified scaling strategy
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                self.cr = 0.6 + 0.4 * (swarm_best - fitness[i]) / (swarm_best - np.min(fitness) + 1e-10)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < swarm_best:
                        swarm_best = trial_fitness

                if evals >= self.budget:
                    break

            if evals % eval_step == 0:
                best_idx = np.argmin(fitness)
                local_search_vector = population[best_idx] + np.random.normal(0, 0.1, self.dim) * (0.5 + 0.5 * np.random.rand())  # Enhanced search
                local_search_vector = np.clip(local_search_vector, self.lower_bound, self.upper_bound)
                local_fitness = func(local_search_vector)
                evals += 1
                if local_fitness < fitness[best_idx]:
                    population[best_idx] = local_search_vector
                    fitness[best_idx] = local_fitness
            
            if np.min(fitness) == best_before:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= self.reinit_threshold:
                worst_indices = fitness.argsort()[-self.pop_size//3:]  # Adjusted fraction
                population[worst_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(worst_indices), self.dim))
                fitness[worst_indices] = np.array([func(ind) for ind in population[worst_indices]])
                evals += len(worst_indices)
                no_improvement_count = 0

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]