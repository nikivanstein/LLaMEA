import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Initial suggested population size
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        eval_step = self.budget // (2 * self.pop_size)  # Dynamic eval step

        while evals < self.budget:
            for i in range(self.pop_size):
                # Mutation and crossover
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
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

            # Dynamic population size adjustment
            if evals % eval_step == 0:
                top_percentage = 0.2  # Keep top 20%
                keep_size = max(2, int(top_percentage * self.pop_size))
                best_indices = fitness.argsort()[:keep_size]
                population = population[best_indices]
                fitness = fitness[best_indices]
                new_members = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size - keep_size, self.dim))
                population = np.vstack((population, new_members))
                fitness = np.concatenate((fitness, np.array([func(ind) for ind in new_members])))
                evals += new_members.shape[0]

        # Return best solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]