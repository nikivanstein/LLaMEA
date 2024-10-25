import numpy as np

class HybridPSO_GA:
    def __init__(self, budget, dim, pop_size=50, omega=0.7, phi_p=0.5, phi_g=0.5, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.F = F
        self.CR = CR

        self.positions = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, np.inf)

        self.global_best_position = None
        self.global_best_score = np.inf

        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, self.positions)
            self.evaluations += self.pop_size

            better_mask = fitness < self.personal_best_scores
            self.personal_best_scores[better_mask] = fitness[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            if np.min(fitness) < self.global_best_score:
                self.global_best_score = np.min(fitness)
                self.global_best_position = self.positions[np.argmin(fitness)]

            inertia_weight = self.omega * (0.5 + 0.5 * np.random.rand()) * (1 - (self.evaluations / self.budget))
            dynamic_phi_g = self.phi_g * np.random.uniform(0.9, 1.1)
            self.velocities = inertia_weight * self.velocities \
                + self.phi_p * np.random.rand(self.pop_size, self.dim) * (self.personal_best_positions - self.positions) \
                + dynamic_phi_g * np.random.rand(self.pop_size, self.dim) * (self.global_best_position - self.positions)

            self.positions += self.velocities
            self.positions = np.clip(self.positions, -5.0, 5.0)

            # Genetic Algorithm Crossover
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    parent1 = self.positions[i]
                    parent2 = self.positions[i + 1]
                    cut_point = np.random.randint(1, self.dim)
                    child1 = np.concatenate((parent1[:cut_point], parent2[cut_point:]))
                    child2 = np.concatenate((parent2[:cut_point], parent1[cut_point:]))
                    child1 = np.clip(child1, -5.0, 5.0)
                    child2 = np.clip(child2, -5.0, 5.0)
                    if func(child1) < fitness[i]:
                        self.positions[i] = child1
                        fitness[i] = func(child1)
                        self.evaluations += 1
                    if func(child2) < fitness[i + 1]:
                        self.positions[i + 1] = child2
                        fitness[i + 1] = func(child2)
                        self.evaluations += 1

        return self.global_best_position, self.global_best_score