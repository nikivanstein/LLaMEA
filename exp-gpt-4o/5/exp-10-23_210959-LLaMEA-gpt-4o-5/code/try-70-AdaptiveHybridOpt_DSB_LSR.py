import numpy as np

class AdaptiveHybridOpt_DSB_LSR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 55  # Slightly increased population for diversity
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.inertia = 0.85  # Adjusted starting inertia

    def pso_update(self):
        cognitive = 1.5  # Further balance between cognitive and social components
        social = 1.9
        
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(2)
            self.velocities[i] = self.inertia * self.velocities[i] + \
                                 cognitive * r1 * (self.personal_best_positions[i] - self.particles[i]) + \
                                 social * r2 * (self.global_best_position - self.particles[i])
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.lower_bound, self.upper_bound)

        # Nonlinear decay with slight randomness for exploration
        self.inertia = max(0.2, self.inertia * np.exp(-self.evaluations/self.budget) + np.random.uniform(-0.05, 0.05))

    def ade_mutation(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = np.random.uniform(0.4, 0.9)  # Broadened mutation factor range for diversity
        return self.personal_best_positions[a] + F * (self.personal_best_positions[b] - self.personal_best_positions[c])
    
    def ade_crossover(self, parent, mutant):
        cross_prob = 0.75 if np.random.rand() > 0.3 else 0.5  # Modified dynamic crossover rate
        cross_points = np.random.rand(self.dim) < cross_prob
        return np.where(cross_points, mutant, parent)
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            # PSO phase with refined update rules
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

            # ADE phase with improved mechanics
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