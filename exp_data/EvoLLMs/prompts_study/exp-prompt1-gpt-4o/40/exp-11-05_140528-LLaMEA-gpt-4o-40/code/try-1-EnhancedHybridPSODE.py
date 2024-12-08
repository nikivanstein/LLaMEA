import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9  # dynamic inertia weight starting value
        self.f_min, self.f_max = 0.5, 0.9  # DE scaling factor range

    def adaptive_inertia_weight(self, evaluations):
        return self.w - 0.4 * (evaluations / self.budget)

    def adaptive_scaling_factor(self, evaluations):
        return self.f_min + (self.f_max - self.f_min) * (1 - evaluations / self.budget)

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = np.copy(self.particles[i])

                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = np.copy(self.particles[i])

            if evaluations >= self.budget:
                break

            # Update velocities and positions for PSO
            self.w = self.adaptive_inertia_weight(evaluations)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.population_size):
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if evaluations < self.budget:
                # Differential Evolution mutation and crossover with adaptive scaling
                f = self.adaptive_scaling_factor(evaluations)
                for i in range(self.population_size):
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self.particles[a] + f * (self.particles[b] - self.particles[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    
                    crossover_mask = np.random.rand(self.dim) < 0.9
                    trial = np.where(crossover_mask, mutant, self.particles[i])
                    
                    trial_score = func(trial)
                    evaluations += 1

                    if trial_score < self.pbest_scores[i]:
                        self.pbest_scores[i] = trial_score
                        self.pbest_positions[i] = trial

                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest_position = trial

                    if evaluations >= self.budget:
                        break

        return self.gbest_score, self.gbest_position