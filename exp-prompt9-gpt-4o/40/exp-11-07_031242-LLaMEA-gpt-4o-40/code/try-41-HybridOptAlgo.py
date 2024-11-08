import numpy as np

class HybridOptAlgo:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.init_pop_size = 40
        self.min_pop_size = 8
        self.f = 0.8  # Mutation factor
        self.cr = 0.9  # Crossover probability
        self.initial_temp = 100.0
        self.temp_decay = 0.94

    def __call__(self, func):
        np.random.seed(0)
        pop_size = self.init_pop_size
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx]
        best_fit = fitness[best_idx]

        def mutation_and_crossover(target_idx, temp):
            indices = list(range(pop_size))
            indices.remove(target_idx)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), *self.bounds)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[target_idx])
            return trial

        temp = self.initial_temp
        while evals < self.budget:
            for i in range(pop_size):
                trial = mutation_and_crossover(i, temp)
                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fit) / temp):
                    pop[i] = trial
                    fitness[i] = trial_fit

                    if trial_fit < best_fit:
                        best_sol = trial
                        best_fit = trial_fit

                temp *= self.temp_decay

                if evals >= self.budget:
                    break

            # Adaptive population size for convergence
            pop_size = max(self.min_pop_size, int(self.init_pop_size * (1 - evals / self.budget)))
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]

        return best_sol, best_fit