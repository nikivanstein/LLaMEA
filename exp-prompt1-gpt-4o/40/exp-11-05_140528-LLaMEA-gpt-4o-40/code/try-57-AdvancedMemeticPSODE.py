import numpy as np
from scipy.special import expit

class AdvancedMemeticPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 50  # Increased population for diversity
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        # Adaptive coefficients with chaotic influence
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_initial, self.w_final = 0.9, 0.4
        self.scale_factor = 0.8  # DE mutation factor
        self.chaos_factor = 0.1  # Chaotic local search influence

    def chaotic_search(self, particle):
        return particle + self.chaos_factor * np.sin(1.0 / (np.abs(particle) + 1e-9))

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            # Calculate progress for dynamic adaptation
            progress = evaluations / self.budget
            self.w = self.w_initial - progress * (self.w_initial - self.w_final)
            
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
                # Differential Evolution mutation and crossover
                for i in range(self.population_size):
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self.particles[a] + self.scale_factor * (self.particles[b] - self.particles[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    
                    crossover_rate = expit(self.pbest_scores[i] - self.gbest_score)
                    crossover_mask = np.random.rand(self.dim) < crossover_rate
                    trial = np.where(crossover_mask, mutant, self.particles[i])
                    
                    # Apply chaotic local search
                    trial = self.chaotic_search(trial)
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