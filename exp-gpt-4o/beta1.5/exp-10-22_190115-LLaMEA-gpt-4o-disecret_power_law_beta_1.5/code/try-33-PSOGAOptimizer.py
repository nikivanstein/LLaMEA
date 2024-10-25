import numpy as np

class PSOGAOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = (-2.0, 2.0)
        self.c1 = 1.7
        self.c2 = 1.3
        self.w = 0.4
        self.ga_crossover_prob = 0.9
        self.ga_mutation_prob = 0.1
        self.adaptive_c1_rate = 0.05
        self.adaptive_c2_rate = 0.07
        self.adaptive_w_rate = 0.02
        self.num_swarms = 4

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(self.velocity_clamp[0], self.velocity_clamp[1], (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([np.inf] * self.pop_size)
        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        swarm_assignments = np.random.choice(self.num_swarms, self.pop_size)

        while evaluations < self.budget:
            scores = np.array([func(p) for p in particles])
            evaluations += self.pop_size

            for i in range(self.pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            self.c1 = max(0.3, self.c1 - self.adaptive_c1_rate * np.random.rand())
            self.c2 = min(2.3, self.c2 + self.adaptive_c2_rate * np.random.rand())
            self.w = max(0.2, self.w - self.adaptive_w_rate * np.random.rand())

            for s in range(self.num_swarms):
                swarm_indices = np.where(swarm_assignments == s)[0]
                if len(swarm_indices) > 0:
                    swarm_global_best_position = personal_best_positions[swarm_indices[np.argmin(personal_best_scores[swarm_indices])]]
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    for i in swarm_indices:
                        cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                        social_component = self.c2 * r2 * (swarm_global_best_position - particles[i])
                        velocities[i] = np.clip(self.w * velocities[i] + cognitive_component + social_component, 
                                                self.velocity_clamp[0], self.velocity_clamp[1])
                        particles[i] = np.clip(particles[i] + velocities[i], self.lower_bound, self.upper_bound)

            for i in range(self.pop_size):
                if np.random.rand() < self.ga_crossover_prob:
                    mates = np.random.choice(self.pop_size, 2, replace=False)
                    beta = np.random.rand()
                    child = beta * particles[mates[0]] + (1 - beta) * particles[mates[1]]
                    if np.random.rand() < self.ga_mutation_prob:
                        mutation_vector = np.random.uniform(-0.5, 0.5, self.dim)
                        child = np.clip(child + mutation_vector, self.lower_bound, self.upper_bound)
                    child_score = func(child)
                    if evaluations < self.budget and child_score < scores[i]:
                        particles[i] = child
                        scores[i] = child_score
                        personal_best_positions[i] = child
                        personal_best_scores[i] = child_score
                        evaluations += 1

        return global_best_position, global_best_score