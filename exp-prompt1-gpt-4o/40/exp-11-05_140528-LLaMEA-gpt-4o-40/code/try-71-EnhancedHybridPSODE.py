import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 50  # Adjusted for better exploration
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))  # Balanced velocity range
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        self.c1_initial, self.c1_final = 2.5, 0.5
        self.c2_initial, self.c2_final = 0.5, 2.5
        self.w_initial, self.w_final = 0.9, 0.4
        self.scale_factor = 0.8  # Adjusted to enhance mutation diversity

    def __call__(self, func):
        evaluations = 0
        cluster_threshold = 0.1  # Threshold for clustering effect
        
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.c1 = self.c1_initial - progress * (self.c1_initial - self.c1_final)
            self.c2 = self.c2_initial + progress * (self.c2_final - self.c2_initial)
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

            # Adaptive PSO with dynamic clustering-based inertia
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            inertia = np.random.uniform(self.w * cluster_threshold, 1.0)
            for i in range(self.population_size):
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest_position - self.particles[i])
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if evaluations < self.budget:
                # Dynamic DE mutation and crossover with multi-scale perturbations
                for i in range(self.population_size):
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self.particles[a] + self.scale_factor * (self.particles[b] - self.particles[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    
                    perturbation = np.random.uniform(-0.1, 0.1, self.dim) * (1 - progress)
                    mutant += perturbation
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    
                    crossover_rate = 0.8
                    crossover_mask = np.random.rand(self.dim) < crossover_rate
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