import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.reinit_threshold = budget // 8

    def __call__(self, func):
        pop_size = self.initial_pop_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = pop_size
        eval_step = self.budget // (4 * pop_size)  # Adjusted dynamic eval step
        no_improvement_count = 0

        while evals < self.budget:
            best_before = np.min(fitness)
            for i in range(pop_size):
                f = 0.5 + 0.3 * np.sin(2 * np.pi * evals / self.budget + np.mean(fitness))  # Adjusted scaling
                indices = np.random.choice(pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + f * (b - c), self.lower_bound, self.upper_bound)
                
                cr = 0.8 + 0.2 * np.exp(-(fitness[i] - np.min(fitness))**2 / (2 * (np.max(fitness) - np.min(fitness) + 1e-10)**2))
                cross_points = np.random.rand(self.dim) < cr
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
                local_search_vector = population[best_idx] + np.random.normal(0, 0.05, self.dim)
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
                worst_indices = fitness.argsort()[-pop_size//5:]  # Adjusted fraction
                population[worst_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(worst_indices), self.dim))
                fitness[worst_indices] = np.array([func(ind) for ind in population[worst_indices]])
                evals += len(worst_indices)
                no_improvement_count = 0

            if evals % (self.budget // 5) == 0:  # Adjust population size dynamically
                pop_size = np.clip(pop_size + np.random.randint(-5, 6), self.dim, 20 * self.dim)
                population = np.vstack((population, np.random.uniform(self.lower_bound, self.upper_bound, (pop_size - population.shape[0], self.dim))))
                fitness = np.append(fitness, [func(ind) for ind in population[-(pop_size - fitness.size):]])
                evals += pop_size - fitness.size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]