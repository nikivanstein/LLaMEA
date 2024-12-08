import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = np.copy(self.particles[0])  # Initialize with a particle
        self.gbest_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        F = 0.8
        CR = 0.9
        
        while self.evaluations < self.budget:
            # Vectorized evaluation of all particles
            scores = np.apply_along_axis(func, 1, self.particles)
            self.evaluations += self.population_size
            
            # Update personal bests
            better_scores_mask = scores < self.pbest_scores
            self.pbest_scores = np.where(better_scores_mask, scores, self.pbest_scores)
            np.copyto(self.pbest_positions, self.particles, where=better_scores_mask[:, None])

            # Update global best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < self.gbest_score:
                self.gbest_score = scores[min_score_idx]
                self.gbest_position = np.copy(self.particles[min_score_idx])

            # Optimize velocities and positions using PSO
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocities = (w * self.velocities + 
                               c1 * r1 * (self.pbest_positions - self.particles) +
                               c2 * r2 * (self.gbest_position - self.particles))
            self.particles += self.velocities
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)
            
            # Adaptive Differential Evolution step
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                
                # Mutation
                candidates = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.particles[candidates]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.particles[i])

                # Selection
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < self.pbest_scores[i]:
                    self.particles[i] = trial
                    self.pbest_scores[i] = trial_score
                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest_position = np.copy(trial)

        return self.gbest_position