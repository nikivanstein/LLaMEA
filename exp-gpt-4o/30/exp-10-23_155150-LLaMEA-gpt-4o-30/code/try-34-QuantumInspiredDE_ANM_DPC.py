import numpy as np

class QuantumInspiredDE_ANM_DPC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.func_evals = 0
        self.alpha = 1.2  # Increased for better exploration
        self.beta = 0.6   # Adjusted for better contraction
        self.gamma = 2.5  # Increased to test more aggressive expansion
        self.base_mutation_factor = 0.8  # Slightly decreased for finer adjustments
        self.base_crossover_rate = 0.8  # Increased for more crossover exploration

    def adaptive_parameters(self, fitness):
        norm_fit = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        mutation_factor = self.base_mutation_factor * (1 - norm_fit)
        crossover_rate = self.base_crossover_rate * (1 - norm_fit**1.5)
        return mutation_factor, crossover_rate

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
                    cross_points[np.random.choice(self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])
                
                f_trial = func(trial)
                self.func_evals += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.pop[i] = trial

            # Dynamic Population Control
            if self.func_evals < self.budget and np.min(self.fitness) < 0.1:
                self.pop_size = max(3, self.pop_size // 2)
                self.pop = self.pop[np.argsort(self.fitness)[:self.pop_size]]
                self.fitness = self.fitness[np.argsort(self.fitness)[:self.pop_size]]

            if self.func_evals < self.budget:
                best_idx = np.argmin(self.fitness)
                simplex = np.array([self.pop[best_idx]] + [self.pop[np.random.randint(self.pop_size)] for _ in range(self.dim)])
                simplex_fitness = np.array([func(p) for p in simplex])
                self.func_evals += len(simplex)

                while self.func_evals < self.budget:
                    order = np.argsort(simplex_fitness)
                    simplex = simplex[order]
                    simplex_fitness = simplex_fitness[order]

                    centroid = np.mean(simplex[:-1], axis=0)
                    reflected = np.clip(centroid + self.alpha * (centroid - simplex[-1]), self.lb, self.ub)
                    f_reflected = func(reflected)
                    self.func_evals += 1

                    if f_reflected < simplex_fitness[0]:
                        expanded = np.clip(centroid + self.gamma * (reflected - centroid), self.lb, self.ub)
                        f_expanded = func(expanded)
                        self.func_evals += 1

                        if f_expanded < f_reflected:
                            simplex[-1] = expanded
                            simplex_fitness[-1] = f_expanded
                        else:
                            simplex[-1] = reflected
                            simplex_fitness[-1] = f_reflected
                    elif f_reflected < simplex_fitness[-2]:
                        simplex[-1] = reflected
                        simplex_fitness[-1] = f_reflected
                    else:
                        contracted = np.clip(centroid + self.beta * (simplex[-1] - centroid), self.lb, self.ub)
                        f_contracted = func(contracted)
                        self.func_evals += 1

                        if f_contracted < simplex_fitness[-1]:
                            simplex[-1] = contracted
                            simplex_fitness[-1] = f_contracted
                        else:
                            for j in range(1, len(simplex)):
                                simplex[j] = np.clip(simplex[0] + 0.5 * (simplex[j] - simplex[0]), self.lb, self.ub)
                                simplex_fitness[j] = func(simplex[j])
                            self.func_evals += len(simplex) - 1

                best_simplex_idx = np.argmin(simplex_fitness)
                if simplex_fitness[best_simplex_idx] < self.fitness[best_idx]:
                    self.pop[best_idx] = simplex[best_simplex_idx]
                    self.fitness[best_idx] = simplex_fitness[best_simplex_idx]

        return self.pop[np.argmin(self.fitness)]