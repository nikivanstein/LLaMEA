import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Initial suggested population size
        self.f = 0.8  # Initial differential weight
        self.cr = 0.9  # Initial crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.reinit_threshold = budget // 10  # Reinitialization threshold

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        eval_step = self.budget // (2 * self.pop_size)  # Dynamic eval step
        no_improvement_count = 0

        while evals < self.budget:
            best_before = np.min(fitness)
            for i in range(self.pop_size):
                # Dynamic mutation factor
                self.f = 0.5 + 0.3 * np.sin(2 * np.pi * evals / self.budget)
                
                # Mutation and crossover
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                # Adaptive crossover probability
                self.cr = 0.5 + 0.5 * (np.max(fitness) - fitness[i]) / (np.max(fitness) - np.min(fitness) + 1e-10)
                
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

            # Local search exploitation
            if evals % eval_step == 0:
                best_idx = np.argmin(fitness)
                local_search_vector = population[best_idx] + np.random.normal(0, 0.1, self.dim)
                local_search_vector = np.clip(local_search_vector, self.lower_bound, self.upper_bound)
                local_fitness = func(local_search_vector)
                evals += 1
                if local_fitness < fitness[best_idx]:
                    population[best_idx] = local_search_vector
                    fitness[best_idx] = local_fitness
            
            # Check for reinitialization condition
            if np.min(fitness) == best_before:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= self.reinit_threshold:
                worst_indices = fitness.argsort()[-self.pop_size//3:]
                population[worst_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(worst_indices), self.dim))
                fitness[worst_indices] = np.array([func(ind) for ind in population[worst_indices]])
                evals += len(worst_indices)
                no_improvement_count = 0

        # Return best solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]