import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 40
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        # Adaptive coefficients with randomness
        self.c1_initial, self.c1_final = 2.0, 1.0
        self.c2_initial, self.c2_final = 1.0, 2.0
        self.w_initial, self.w_final = 0.8, 0.3
        self.base_scale_factor = 0.6

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.c1 = self.c1_initial + progress * (self.c1_final - self.c1_initial)
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

            # Update velocities and positions for adaptive PSO
            for i in range(self.population_size):
                neighbors = np.random.choice(self.population_size, 3, replace=False)
                best_neighbor = min(neighbors, key=lambda x: self.pbest_scores[x])
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.pbest_positions[best_neighbor] - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Differential Evolution with self-adaptive mutation factor
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutation_factor = self.base_scale_factor + 0.1 * np.random.rand()
                mutant = self.particles[a] + mutation_factor * (self.particles[b] - self.particles[c])
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