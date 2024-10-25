import numpy as np

class HybridCooperativeStrategyOpt_AIDE:
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
        cognitive = 1.5  # Slightly adjusted for improved exploration
        social = 1.9
        
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(2)
            self.velocities[i] = self.inertia * self.velocities[i] + \
                                 cognitive * r1 * (self.personal_best_positions[i] - self.particles[i]) + \
                                 social * r2 * (self.global_best_position - self.particles[i])
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.lower_bound, self.upper_bound)

        # Adaptive inertia decay for balanced exploration and exploitation
        self.inertia = max(0.4, self.inertia * np.exp(-0.05 * self.evaluations/self.budget))

    def de_mutation(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = np.random.uniform(0.3, 0.7)  # Robust mutation factor for diversity
        return self.personal_best_positions[a] + F * (self.personal_best_positions[b] - self.personal_best_positions[c])
    
    def de_crossover(self, parent, mutant):
        cross_prob = 0.8 if np.random.rand() > 0.1 else 0.5  # Dynamic crossover rate for adaptability
        cross_points = np.random.rand(self.dim) < cross_prob
        return np.where(cross_points, mutant, parent)
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            # PSO phase with improved balance between exploration and exploitation
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

            # DE phase with refined mutation and crossover strategies
            for i in range(self.pop_size):
                mutant = self.de_mutation(i)
                trial = self.de_crossover(self.particles[i], mutant)
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