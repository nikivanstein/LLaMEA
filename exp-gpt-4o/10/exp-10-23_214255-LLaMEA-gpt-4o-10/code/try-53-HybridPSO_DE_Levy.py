import numpy as np

class HybridPSO_DE_Levy:
    def __init__(self, budget, dim, pop_size=50, omega=0.5, phi_p=0.5, phi_g=0.5, F=0.8, CR=0.9):
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
        self.subpopulations = np.array_split(self.positions, 2)

    def levy_flight(self, dim):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        while self.evaluations < self.budget:
            for subpop in self.subpopulations:
                fitness = np.apply_along_axis(func, 1, subpop)
                self.evaluations += len(subpop)

                better_mask = fitness < self.personal_best_scores[:len(subpop)]
                self.personal_best_scores[:len(subpop)][better_mask] = fitness[better_mask]
                self.personal_best_positions[:len(subpop)][better_mask] = subpop[better_mask]

                if np.min(fitness) < self.global_best_score:
                    self.global_best_score = np.min(fitness)
                    self.global_best_position = subpop[np.argmin(fitness)]

                inertia_weight = self.omega * np.random.uniform(0.5, 1.0) * (1 - (self.evaluations / self.budget))
                dynamic_phi_g = self.phi_g * (np.random.uniform(0.9, 1.1) if self.evaluations < self.budget / 2 else np.random.uniform(0.8, 1.2))
                self.velocities[:len(subpop)] = inertia_weight * self.velocities[:len(subpop)] \
                    + self.phi_p * np.random.rand(len(subpop), self.dim) * (self.personal_best_positions[:len(subpop)] - subpop) \
                    + dynamic_phi_g * np.random.rand(len(subpop), self.dim) * (self.global_best_position - subpop)

                subpop += self.velocities[:len(subpop)]
                self.subpopulations = np.array_split(self.positions, 2)

                for i in range(len(subpop)):
                    indices = np.random.choice(np.delete(np.arange(len(subpop)), i), 3, replace=False)
                    mutant = subpop[indices[0]] + self.F * (subpop[indices[1]] - subpop[indices[2]])
                    mutant = np.clip(mutant, -5.0, 5.0)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, subpop[i])
                    trial_score = func(trial)
                    self.evaluations += 1
                    if trial_score < fitness[i]:
                        subpop[i] = trial
                        fitness[i] = trial_score
                        if trial_score < self.personal_best_scores[i]:
                            self.personal_best_scores[i] = trial_score
                            self.personal_best_positions[i] = trial
                    else:
                        levy_step = self.levy_flight(self.dim)
                        candidate = subpop[i] + levy_step
                        candidate = np.clip(candidate, -5.0, 5.0)
                        candidate_score = func(candidate)
                        self.evaluations += 1
                        if candidate_score < fitness[i]:
                            subpop[i] = candidate
                            fitness[i] = candidate_score
                            if candidate_score < self.personal_best_scores[i]:
                                self.personal_best_scores[i] = candidate_score
                                self.personal_best_positions[i] = candidate

                subpop[:] = np.clip(subpop, -5.0, 5.0)

        return self.global_best_position, self.global_best_score