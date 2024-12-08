import numpy as np

class AdaptiveMemeticDE_GR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.func_evals = 0
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 2.0
        self.base_mutation_factor = 0.9
        self.base_crossover_rate = 0.8

    def adaptive_parameters(self, fitness):
        norm_fit = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        mutation_factor = self.base_mutation_factor * (1 - norm_fit)
        crossover_rate = self.base_crossover_rate * norm_fit
        return mutation_factor, crossover_rate

    def gradient_based_refinement(self, x, func):
        step_size = 1e-2
        gradient = np.array([(func(x + step_size * np.eye(1, self.dim, i)[0]) - func(x)) / step_size for i in range(self.dim)])
        x_new = x - step_size * gradient
        x_new = np.clip(x_new, self.lb, self.ub)
        return x_new

    def __call__(self, func):
        for i in range(self.pop_size):
            self.fitness[i] = func(self.pop[i])
            self.func_evals += 1
            if self.func_evals >= self.budget:
                return self.pop[np.argmin(self.fitness)]

        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    return self.pop[np.argmin(self.fitness)]

                mutation_factor, crossover_rate = self.adaptive_parameters(self.fitness)
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.pop[indices]
                mutant = np.clip(a + mutation_factor[i] * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < crossover_rate[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])
                
                f_trial = func(trial)
                self.func_evals += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.pop[i] = trial

            if self.func_evals < self.budget:
                best_idx = np.argmin(self.fitness)
                refined = self.gradient_based_refinement(self.pop[best_idx], func)
                f_refined = func(refined)
                self.func_evals += 1

                if f_refined < self.fitness[best_idx]:
                    self.fitness[best_idx] = f_refined
                    self.pop[best_idx] = refined

        return self.pop[np.argmin(self.fitness)]