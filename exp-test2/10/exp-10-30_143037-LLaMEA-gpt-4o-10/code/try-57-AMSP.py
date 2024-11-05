import numpy as np

class AMSP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.num_swarms = 3
        self.inertia = 0.7
        self.cognitive = 1.4
        self.social = 1.4
        self.positions = [np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim)) 
                          for _ in range(self.num_swarms)]
        self.velocities = [np.zeros((self.num_particles, self.dim)) for _ in range(self.num_swarms)]
        self.local_best_positions = [pos.copy() for pos in self.positions]
        self.local_best_scores = [np.full(self.num_particles, np.inf) for _ in range(self.num_swarms)]
        self.global_best_score = np.inf
        self.global_best_position = np.zeros(self.dim)
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for swarm_index in range(self.num_swarms):
                for i in range(self.num_particles):
                    score = func(self.positions[swarm_index][i])
                    self.evaluations += 1

                    if score < self.local_best_scores[swarm_index][i]:
                        self.local_best_scores[swarm_index][i] = score
                        self.local_best_positions[swarm_index][i] = self.positions[swarm_index][i].copy()

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[swarm_index][i].copy()

            for swarm_index in range(self.num_swarms):
                r1, r2 = np.random.rand(2, self.num_particles, self.dim)
                cognitive_component = self.cognitive * r1 * (self.local_best_positions[swarm_index] - self.positions[swarm_index])
                social_component = self.social * r2 * (self.global_best_position - self.positions[swarm_index])
                self.velocities[swarm_index] = (self.inertia * self.velocities[swarm_index] + cognitive_component + social_component)

                # Dynamic adjustment of cognitive and social factors
                if self.evaluations > 0.5 * self.budget: 
                    self.cognitive = 1.3 + 0.2 * np.random.rand()
                    self.social = 1.1 + 0.3 * np.random.rand()
                self.positions[swarm_index] += self.velocities[swarm_index]
                self.positions[swarm_index] = np.clip(self.positions[swarm_index], self.lower_bound, self.upper_bound)

                # Selective swarm collaboration
                if swarm_index == np.argmin([np.min(self.local_best_scores[swarm]) for swarm in range(self.num_swarms)]):
                    self.positions[swarm_index] += 0.1 * np.random.rand(self.num_particles, self.dim) * (self.positions[(swarm_index+1) % self.num_swarms] - self.positions[swarm_index])

        return self.global_best_position, self.global_best_score