import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 100
        self.swarm_size = self.initial_swarm_size
        self.position = np.random.uniform(-5.0, 5.0, (self.swarm_size, dim))
        self.velocity = np.zeros((self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.phi = 0.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.learning_rate_decay = 0.98  # Adjusted adaptive learning rate decay
        self.chaos_factor = 0.2  # Chaotic influence factor
        self.opposition_factor = 0.5  # Opposition factor for better exploration

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
            self.velocity = (inertia_weight * self.velocity + cognitive_component + social_component)

            # Chaos-enhanced exploration
            chaos_component = self.chaos_factor * (np.random.rand(self.swarm_size, self.dim) - 0.5)
            new_position = self.position + self.velocity + chaos_component

            # Opposition-based learning
            opposition_position = 5.0 - new_position
            opposition_fitness = np.array([func(pos) for pos in opposition_position])
            evals += self.swarm_size

            improved_mask = opposition_fitness < fitness_values
            new_position[improved_mask] = opposition_position[improved_mask]

            # Enforce boundaries
            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position
            
            # Dynamic swarm size reduction
            if evals > self.budget * 0.5:
                self.swarm_size = int(self.initial_swarm_size * 0.5)
                self.position = self.position[:self.swarm_size]
                self.velocity = self.velocity[:self.swarm_size]
                self.personal_best_position = self.personal_best_position[:self.swarm_size]
                self.personal_best_value = self.personal_best_value[:self.swarm_size]
            
        return self.global_best_position, self.global_best_value