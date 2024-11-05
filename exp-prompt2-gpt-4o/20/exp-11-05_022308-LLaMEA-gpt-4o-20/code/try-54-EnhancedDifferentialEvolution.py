import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.reinit_threshold = budget // 10
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.initial_pop_size
        eval_step = self.budget // (4 * self.initial_pop_size)
        no_improvement_count = 0
        pop_size_variation = int(self.initial_pop_size * 0.2)  # Dynamically adjust population size

        while evals < self.budget:
            best_before = np.min(fitness)
            for i in range(self.initial_pop_size):
                # Dynamic scaling factor based on diversity
                current_scaling = self.scaling_factor * (1.0 - np.std(fitness) / np.mean(fitness))
                indices = np.random.choice(self.initial_pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + current_scaling * (b - c), self.lower_bound, self.upper_bound)
                
                self.crossover_rate = 0.6 + 0.4 * (1 - fitness[i] / (np.max(fitness) + 1e-10))
                cross_points = np.random.rand(self.dim) < self.crossover_rate
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

            if evals % eval_step == 0:
                best_idx = np.argmin(fitness)
                local_search_vector = population[best_idx] + np.random.normal(0, 0.1, self.dim)
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
                worst_indices = fitness.argsort()[-self.initial_pop_size//5:]
                population[worst_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(worst_indices), self.dim))
                fitness[worst_indices] = np.array([func(ind) for ind in population[worst_indices]])
                evals += len(worst_indices)
                no_improvement_count = 0

            # Dynamic population resizing
            if no_improvement_count % (self.reinit_threshold // 2) == 0:
                if evals < self.budget * 0.5:
                    self.initial_pop_size += pop_size_variation
                else:
                    self.initial_pop_size -= pop_size_variation
                population = np.resize(population, (self.initial_pop_size, self.dim))
                fitness = np.resize(fitness, self.initial_pop_size)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]