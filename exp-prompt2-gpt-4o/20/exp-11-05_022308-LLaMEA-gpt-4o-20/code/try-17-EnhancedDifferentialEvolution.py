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

    def chaos_mutation(self, x):
        # Logistic map for chaos-based mutation
        a = 4.0  # Logistic map parameter
        return self.lower_bound + (self.upper_bound - self.lower_bound) * a * x * (1 - x)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        eval_step = self.budget // (2 * self.pop_size)

        while evals < self.budget:
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                
                # Chaos-based mutation
                chaotic_ratio = np.random.rand()
                mutant = np.clip(a + self.f * (b - c) + chaotic_ratio * (self.chaos_mutation(a) - a), self.lower_bound, self.upper_bound)
                
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

            if evals % eval_step == 0:
                top_percentage = 0.2
                keep_size = max(2, int(top_percentage * self.pop_size))
                best_indices = fitness.argsort()[:keep_size]
                population = population[best_indices]
                fitness = fitness[best_indices]
                new_members = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size - keep_size, self.dim))
                population = np.vstack((population, new_members))
                fitness = np.concatenate((fitness, np.array([func(ind) for ind in new_members])))
                evals += new_members.shape[0]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]