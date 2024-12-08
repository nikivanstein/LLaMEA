import numpy as np

class QEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40 + (dim * 4)  # Adjusted swarm size for faster convergence
        self.position = np.random.uniform(-5.0, 5.0, (self.swarm_size, dim))
        self.velocity = np.zeros((self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.w_max = 0.8
        self.w_min = 0.3
        self.c1 = 2.5  # Increased cognitive component for local exploration
        self.c2 = 2.5  # Increased social component for global exploration
        self.chaos_factor = 0.2  # Reduced chaotic influence to stabilize convergence
        self.reinit_prob = 0.1  # Probability of reinitializing particles

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            # Evaluate fitness
            fitness_values = np.array([func(pos) for pos in self.position])
            evals += self.swarm_size

            # Update personal bests
            better_mask = fitness_values < self.personal_best_value
            self.personal_best_value[better_mask] = fitness_values[better_mask]
            self.personal_best_position[better_mask] = self.position[better_mask]

            # Update global best
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < self.global_best_value:
                self.global_best_value = fitness_values[min_fitness_idx]
                self.global_best_position = self.position[min_fitness_idx]

            # Update velocities and positions
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            
            cognitive_component = self.c1 * r1 * (self.personal_best_position - self.position)
            social_component = self.c2 * r2 * (self.global_best_position - self.position)
            self.velocity = inertia_weight * (self.velocity + cognitive_component + social_component)

            # Phase-based velocity adjustment
            if evals % (self.budget // 3) == 0:
                self.velocity *= 0.5

            # Stochastic reinitialization
            reinit_mask = np.random.rand(self.swarm_size) < self.reinit_prob
            reinit_positions = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            self.position = np.where(reinit_mask[:, None], reinit_positions, self.position)

            # Update position with chaos-enhanced exploration
            chaos_component = self.chaos_factor * (np.random.rand(self.swarm_size, self.dim) - 0.5)
            new_position = self.position + self.velocity + chaos_component

            # Enforce boundaries
            self.position = np.clip(new_position, -5.0, 5.0)

        return self.global_best_position, self.global_best_value