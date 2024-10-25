import numpy as np

class AdaptiveMultiStrategySwarmOpt_DIS_FC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 50
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.inertia = 0.9

    def pso_update(self):
        cognitive = 1.5  # Slight adjustment for better exploratory behavior
        social = 1.7
        
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(2)
            self.velocities[i] = self.inertia * self.velocities[i] + \
                                 cognitive * r1 * (self.personal_best_positions[i] - self.particles[i]) + \
                                 social * r2 * (self.global_best_position - self.particles[i])
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.lower_bound, self.upper_bound)

        # Dynamic inertia scaled by function evaluations for improved convergence
        self.inertia = max(0.3, 0.9 - (0.6 * self.evaluations/self.budget))

    def fuzzy_mutation(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = np.random.uniform(0.4, 0.9)  # Adjusted mutation factor for diversity
        return self.personal_best_positions[a] + F * (self.personal_best_positions[b] - self.personal_best_positions[c])
    
    def fuzzy_crossover(self, parent, mutant):
        cross_prob = 0.85  # Static crossover rate for simplicity
        blend_rate = np.random.rand(self.dim) * (0.5 if np.random.rand() > 0.5 else 0.3)  # Blending for diversity
        cross_points = np.random.rand(self.dim) < cross_prob
        return np.where(cross_points, blend_rate * mutant + (1 - blend_rate) * parent, parent)
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            # PSO phase with adaptive inertia
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

            # Fuzzy mutation phase for diverse exploration
            for i in range(self.pop_size):
                mutant = self.fuzzy_mutation(i)
                trial = self.fuzzy_crossover(self.particles[i], mutant)
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