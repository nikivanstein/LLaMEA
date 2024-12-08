import numpy as np

class AdaptiveQuantumPSO_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = max(20, 5 * dim)  # Increase population size for better diversity
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, dim))
        self.personal_best = np.copy(self.pop)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.func_evals = 0
        self.alpha = 1.2  # Adjust reflection parameter for more aggressive search
        self.beta = 0.4  # Adjust contraction parameter for better convergence
        self.gamma = 1.5  # Adjust expansion parameter for higher exploration
        self.inertia_weight = 0.8  # Increase inertia to maintain momentum
        self.cognitive_weight = 1.3  # Balance cognitive weight
        self.social_weight = 1.7  # Increase social weight for better convergence

    def update_particle(self, i, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                              self.cognitive_weight * r1 * (self.personal_best[i] - self.pop[i]) +
                              self.social_weight * r2 * (self.global_best - self.pop[i]))
        self.pop[i] = np.clip(self.pop[i] + self.velocities[i], self.lb, self.ub)
        f_value = func(self.pop[i])
        self.func_evals += 1
        if f_value < self.personal_best_fitness[i]:
            self.personal_best_fitness[i] = f_value
            self.personal_best[i] = self.pop[i]
        if f_value < self.global_best_fitness:
            self.global_best_fitness = f_value
            self.global_best = self.pop[i]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return np.clip(child, self.lb, self.ub)

    def __call__(self, func):
        # Initial evaluation
        for i in range(self.pop_size):
            f_value = func(self.pop[i])
            self.func_evals += 1
            self.personal_best_fitness[i] = f_value
            self.personal_best[i] = self.pop[i]
            if f_value < self.global_best_fitness:
                self.global_best_fitness = f_value
                self.global_best = self.pop[i]

        while self.func_evals < self.budget:
            # Update particles
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    return self.global_best
                self.update_particle(i, func)

            # Hybrid optimization with Crossover and Nelder-Mead
            if self.func_evals < self.budget:
                best_idx = np.argmin(self.personal_best_fitness)
                simplex = np.array([self.personal_best[best_idx]] + [self.personal_best[np.random.randint(self.pop_size)] for _ in range(self.dim)])
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

                # Crossover for further refinement
                parent1, parent2 = simplex[np.argmin(simplex_fitness)], simplex[np.argmax(simplex_fitness)]
                child = self.crossover(parent1, parent2)
                f_child = func(child)
                self.func_evals += 1

                if f_child < np.max(simplex_fitness):
                    simplex[np.argmax(simplex_fitness)] = child
                    simplex_fitness[np.argmax(simplex_fitness)] = f_child

                best_simplex_idx = np.argmin(simplex_fitness)
                if simplex_fitness[best_simplex_idx] < self.personal_best_fitness[best_idx]:
                    self.personal_best[best_idx] = simplex[best_simplex_idx]
                    self.personal_best_fitness[best_idx] = simplex_fitness[best_simplex_idx]

        return self.global_best