import numpy as np

class AdaptiveMultiPSO_DE_Improved:
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

    def __call__(self, func):
        subpop_count = 3  # Changed number of subpopulations for diversity
        while self.evaluations < self.budget:
            # Dynamically adjust subpopulations based on evaluations
            subpop_size = max(self.pop_size // subpop_count, 1)
            subpopulations = [self.positions[i:i + subpop_size] for i in range(0, self.pop_size, subpop_size)]
            
            for subpop in subpopulations:
                fitness = np.apply_along_axis(func, 1, subpop)
                self.evaluations += len(subpop)

                better_mask = fitness < self.personal_best_scores[:len(subpop)]
                self.personal_best_scores[:len(subpop)][better_mask] = fitness[better_mask]
                self.personal_best_positions[:len(subpop)][better_mask] = subpop[better_mask]

                if np.min(fitness) < self.global_best_score:
                    self.global_best_score = np.min(fitness)
                    self.global_best_position = subpop[np.argmin(fitness)]

                inertia_weight = self.omega * (0.7298 if self.evaluations < self.budget * 0.75 else 0.7)  # Reduced stochastic element
                dynamic_phi_g = self.phi_g * (np.random.uniform(0.95, 1.05) if self.evaluations < self.budget / 2 else np.random.uniform(0.85, 1.15))
                self.velocities[:len(subpop)] = inertia_weight * self.velocities[:len(subpop)] \
                    + self.phi_p * np.random.rand(len(subpop), self.dim) * (self.personal_best_positions[:len(subpop)] - subpop) \
                    + dynamic_phi_g * np.random.rand(len(subpop), self.dim) * (self.global_best_position - subpop)

                subpop += self.velocities[:len(subpop)]

                # Differential Evolution Mutation and Crossover
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

                # Ensure particles do not exceed search space boundaries with stochastic boundary correction
                exceed_mask = (subpop < -5.0) | (subpop > 5.0)
                subpop[exceed_mask] = np.random.uniform(-5.0, 5.0, subpop[exceed_mask].shape)
                
        return self.global_best_position, self.global_best_score