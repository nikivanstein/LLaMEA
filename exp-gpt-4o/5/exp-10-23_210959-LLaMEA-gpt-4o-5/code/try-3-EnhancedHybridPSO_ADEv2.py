import numpy as np

class EnhancedHybridPSO_ADEv2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 60  # Increased population size for better diversity
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.inertia = 0.9  # Adaptive inertia for PSO

    def pso_update(self):
        cognitive = 1.5
        social = 1.5
        r1_array, r2_array = np.random.rand(2, self.pop_size, self.dim)
        
        self.velocities = self.inertia * self.velocities + \
                          cognitive * r1_array * (self.personal_best_positions - self.particles) + \
                          social * r2_array * (self.global_best_position - self.particles)
        self.particles = np.clip(self.particles + self.velocities, self.lower_bound, self.upper_bound)

        # Adaptive inertia decay
        self.inertia = max(0.4, self.inertia * 0.98)

    def ade_mutation(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = np.random.uniform(0.5, 1.0)  # Dynamic scaling factor
        return self.personal_best_positions[a] + F * (self.personal_best_positions[b] - self.personal_best_positions[c])
    
    def ade_crossover(self, parent, mutant):
        cross_points = np.random.rand(self.dim) < 0.9  # Increased crossover rate
        return np.where(cross_points, mutant, parent)
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            # PSO phase
            self.pso_update()

            # Evaluate PSO particles
            for i in range(self.pop_size):
                score = func(self.particles[i])
                self.evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            if self.evaluations >= self.budget:
                break

            # ADE phase
            for i in range(self.pop_size):
                mutant = self.ade_mutation(i)
                trial = self.ade_crossover(self.particles[i], mutant)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                score = func(trial)
                self.evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = trial

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = trial

        return self.global_best_position