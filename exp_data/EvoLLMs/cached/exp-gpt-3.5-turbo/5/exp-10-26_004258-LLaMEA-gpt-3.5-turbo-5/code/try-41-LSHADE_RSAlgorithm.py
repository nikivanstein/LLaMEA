import numpy as np
from scipy.optimize import differential_evolution

class LSHADE_RSAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        def optimize_lshade_rs(x0):
            bounds = [(-5, 5)] * self.dim
            result_de = differential_evolution(objective, bounds, maxiter=self.budget, seed=42, popsize=10, tol=0.01)

            population_size = 20
            memory_size = 5
            p_best = 0.11
            scaling_factor = 0.7
            crossover_prob = 0.9
            restart_threshold = 0.7

            population = np.random.uniform(-5, 5, (population_size, self.dim))
            archive = []
            arc_func_values = np.zeros(memory_size) + np.inf
            best_position = population[0]
            best_value = func(population[0])

            for _ in range(self.budget):
                for i in range(population_size):
                    idxs = np.random.choice(population_size, 5, replace=False)
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    mutant = population[idxs[0]] + scaling_factor * (population[idxs[1]] - population[idxs[2]] + population[idxs[3]] - population[idxs[4]])
                    crossover_mask = np.random.rand(self.dim) < crossover_prob
                    trial = np.where(crossover_mask, mutant, population[i])

                    trial_value = func(trial)
                    if trial_value < func(population[i]):
                        population[i] = trial
                        if trial_value < best_value:
                            best_value = trial_value
                            best_position = trial

                    if trial_value < arc_func_values[0]:
                        idx = np.argmax(arc_func_values)
                        arc_func_values[idx] = trial_value
                        archive[idx] = trial
                        if np.random.rand() < p_best:
                            idx = np.random.randint(0, memory_size)
                            population[i] = archive[idx]

                if np.mean(arc_func_values) < restart_threshold * best_value:
                    population = np.random.uniform(-5, 5, (population_size, self.dim))

            return best_position, best_value

        x0 = np.random.uniform(-5, 5, self.dim)

        return optimize_lshade_rs(x0)