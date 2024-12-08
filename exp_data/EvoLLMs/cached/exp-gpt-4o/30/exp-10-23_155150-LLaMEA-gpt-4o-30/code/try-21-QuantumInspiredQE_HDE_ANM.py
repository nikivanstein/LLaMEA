import numpy as np

class QuantumInspiredQE_HDE_ANM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 8 * dim  # Adjusted population size for quantum effects
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.func_evals = 0
        self.alpha = 1.3  # Modified alpha for better reflection
        self.beta = 0.4   # Adjusted contraction parameter
        self.gamma = 2.1  # Enhanced expansion parameter
        self.base_mutation_factor = 0.7  # Slight mutation factor reduction
        self.base_crossover_rate = 0.85  # Adjusted crossover rate
        self.quantum_factor = 0.1  # New quantum factor for state influence

    def adaptive_parameters(self, fitness):
        norm_fit = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        mutation_factor = self.base_mutation_factor * (1 - norm_fit)
        crossover_rate = self.base_crossover_rate * (1 - norm_fit)
        return mutation_factor, crossover_rate

    def quantum_superposition(self):
        # Generate new states based on quantum superposition principles
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim)) * self.quantum_factor

    def __call__(self, func):
        for i in range(self.pop_size):
            self.fitness[i] = func(self.pop[i])
            self.func_evals += 1
            if self.func_evals >= self.budget:
                return self.pop[np.argmin(self.fitness)]

        while self.func_evals < self.budget:
            # Differential Evolution with adaptive parameters
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

            # Quantum-inspired update
            if self.func_evals < self.budget:
                new_quantum_states = self.quantum_superposition()
                for q_state in new_quantum_states:
                    if self.func_evals >= self.budget:
                        break
                    f_q_state = func(q_state)
                    self.func_evals += 1
                    max_fit_idx = np.argmax(self.fitness)
                    if f_q_state < self.fitness[max_fit_idx]:
                        self.fitness[max_fit_idx] = f_q_state
                        self.pop[max_fit_idx] = q_state

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

                best_simplex_idx = np.argmin(simplex_fitness)
                if simplex_fitness[best_simplex_idx] < self.fitness[best_idx]:
                    self.pop[best_idx] = simplex[best_simplex_idx]
                    self.fitness[best_idx] = simplex_fitness[best_simplex_idx]

        return self.pop[np.argmin(self.fitness)]