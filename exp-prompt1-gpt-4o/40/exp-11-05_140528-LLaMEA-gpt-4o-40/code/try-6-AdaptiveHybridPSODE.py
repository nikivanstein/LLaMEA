import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 50  # Further increased population for diversity
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        # Adaptive coefficients with sigmoid modulation for smoother adaptation
        self.c1_initial, self.c1_final = 2.5, 0.5
        self.c2_initial, self.c2_final = 0.5, 2.5
        self.w_initial, self.w_final = 0.9, 0.4
        self.scale_factor_min, self.scale_factor_max = 0.6, 1.0  # DE mutation factor range

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            # Calculate adaptive parameters using a sigmoid function
            progress = evaluations / self.budget
            sig_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            self.c1 = self.c1_initial - sig_progress * (self.c1_initial - self.c1_final)
            self.c2 = self.c2_initial + sig_progress * (self.c2_final - self.c2_initial)
            self.w = self.w_initial - sig_progress * (self.w_initial - self.w_final)
            self.scale_factor = self.scale_factor_min + sig_progress * (self.scale_factor_max - self.scale_factor_min)
            
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.particles[i]

                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.particles[i]

                if evaluations >= self.budget:
                    break

            # Update velocities and positions for adaptive PSO
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.population_size):
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if evaluations < self.budget:
                # Differential Evolution mutation and crossover with adaptive mutation factor
                for i in range(self.population_size):
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self.particles[a] + self.scale_factor * (self.particles[b] - self.particles[c])
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