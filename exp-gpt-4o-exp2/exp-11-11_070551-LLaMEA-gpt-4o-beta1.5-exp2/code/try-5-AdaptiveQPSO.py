import numpy as np

class AdaptiveQPSO:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.7, cognitive_param=1.5, social_param=1.5, quantum_param=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_param = cognitive_param
        self.social_param = social_param
        self.quantum_param = quantum_param
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (num_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(num_particles, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_score = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            evaluations += self.num_particles

            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            exploration_factor = (1 - (evaluations / self.budget)) ** 2
            mean_position = np.mean(self.positions, axis=0)
            for i in range(self.num_particles):
                phi = np.random.rand(self.dim)
                new_position = (1 - self.quantum_param) * self.positions[i] + self.quantum_param * mean_position + phi * (self.personal_best_positions[i] - self.positions[i])
                self.positions[i] = np.clip(new_position, self.lower_bound, self.upper_bound)

            self.velocities = (
                self.inertia_weight * self.velocities +
                self.cognitive_param * np.random.rand(self.num_particles, self.dim) * (self.personal_best_positions - self.positions) +
                self.social_param * np.random.rand(self.num_particles, self.dim) * (self.global_best_position - self.positions)
            )
            self.velocities = exploration_factor * self.velocities
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score