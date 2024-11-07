import numpy as np

class AdaptiveMultiSwarmPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 50  # Increased population for diversity
        self.sub_pop_size = 10  # Divide into sub-swarms
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        self.c1_initial, self.c1_final = 2.0, 0.5
        self.c2_initial, self.c2_final = 0.5, 2.0
        self.w_initial, self.w_final = 0.9, 0.4
        self.scale_factor = 0.8  # Adjusted DE factor
        self.crossover_rate = 0.9

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.c1 = self.c1_initial - progress * (self.c1_initial - self.c1_final)
            self.c2 = self.c2_initial + progress * (self.c2_final - self.c2_initial)
            self.w = self.w_initial - progress * (self.w_initial - self.w_final)
            
            # Divide population into sub-swarms for enhanced exploration
            for sub_idx in range(0, self.population_size, self.sub_pop_size):
                sub_particles = slice(sub_idx, sub_idx + self.sub_pop_size)
                best_sub_score = np.inf
                best_sub_position = None
                
                for i in range(sub_particles.start, sub_particles.stop):
                    score = func(self.particles[i])
                    evaluations += 1

                    if score < self.pbest_scores[i]:
                        self.pbest_scores[i] = score
                        self.pbest_positions[i] = self.particles[i]

                    if score < best_sub_score:
                        best_sub_score = score
                        best_sub_position = self.particles[i]
                    
                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = self.particles[i]

                    if evaluations >= self.budget:
                        return self.gbest_score, self.gbest_position

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = np.random.uniform(0.5, 1.0)
                for i in range(sub_particles.start, sub_particles.stop):
                    cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                    social = self.c2 * r2 * (best_sub_position - self.particles[i])
                    self.velocities[i] = inertia * self.w * self.velocities[i] + cognitive + social
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
            
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.particles[a] + self.scale_factor * (self.particles[b] - self.particles[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
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
                    return self.gbest_score, self.gbest_position

        return self.gbest_score, self.gbest_position