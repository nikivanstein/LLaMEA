import numpy as np

class AdaptiveHybridDE_ANM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim  # Increased population size for better diversity
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.func_evals = 0
        self.alpha = 1.2  # Increased reflection parameter for faster exploration
        self.beta = 0.5  # Further modified contraction factor for flexibility
        self.gamma = 2.5  # Increased expansion coefficient to enhance global search
        self.mutation_factor = 0.8  # Adaptive mutation factor based on success history
        self.crossover_rate = 0.9  # Enhanced crossover rate for more aggressive mixing

    def __call__(self, func):
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = func(self.pop[i])
            self.func_evals += 1
            if self.func_evals >= self.budget:
                return self.pop[np.argmin(self.fitness)]

        success_history = []

        while self.func_evals < self.budget:
            # Differential Evolution
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    return self.pop[np.argmin(self.fitness)]

                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.pop[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])
                
                f_trial = func(trial)
                self.func_evals += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.pop[i] = trial
                    success_history.append(1)
                else:
                    success_history.append(0)

            # Update mutation factor adaptively
            if len(success_history) > 50:
                success_rate = np.mean(success_history[-50:])
                self.mutation_factor = 0.5 + success_rate * 0.5

            # Adaptive Nelder-Mead
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
                    reflected = centroid + self.alpha * (centroid - simplex[-1])
                    reflected = np.clip(reflected, self.lb, self.ub)
                    f_reflected = func(reflected)
                    self.func_evals += 1

                    if f_reflected < simplex_fitness[0]:
                        expanded = centroid + self.gamma * (reflected - centroid)
                        expanded = np.clip(expanded, self.lb, self.ub)
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
                        contracted = centroid + self.beta * (simplex[-1] - centroid)
                        contracted = np.clip(contracted, self.lb, self.ub)
                        f_contracted = func(contracted)
                        self.func_evals += 1

                        if f_contracted < simplex_fitness[-1]:
                            simplex[-1] = contracted
                            simplex_fitness[-1] = f_contracted
                        else:
                            for j in range(1, len(simplex)):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], self.lb, self.ub)
                                simplex_fitness[j] = func(simplex[j])
                            self.func_evals += len(simplex) - 1

                # Update population with the best simplex solution
                best_simplex_idx = np.argmin(simplex_fitness)
                if simplex_fitness[best_simplex_idx] < self.fitness[best_idx]:
                    self.pop[best_idx] = simplex[best_simplex_idx]
                    self.fitness[best_idx] = simplex_fitness[best_simplex_idx]

        return self.pop[np.argmin(self.fitness)]