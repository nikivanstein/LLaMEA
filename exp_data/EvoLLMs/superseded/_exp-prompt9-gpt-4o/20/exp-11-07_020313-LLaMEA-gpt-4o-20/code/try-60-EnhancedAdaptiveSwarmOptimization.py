import numpy as np

class EnhancedAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 30  # Slightly reduced swarm size
        self.alpha_cognitive = 0.4  # Increased personal influence
        self.alpha_social = 0.3  # Further increased social influence
        self.inertia_start = 0.8  # Adjusted starting inertia
        self.inertia_end = 0.3  # Modified ending inertia
        self.mutation_prob = 0.15  # Increased mutation probability
        self.crossover_prob = 0.4  # Higher crossover probability

    def __call__(self, func):
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        particle_velocities = np.random.uniform(-0.6, 0.6, (self.swarm_size, self.dim))  # Modified velocity range
        personal_best_positions = np.copy(particle_positions)
        personal_best_values = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_end + (self.inertia_start - self.inertia_end) * (1 - evaluations / self.budget)
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                rand_cognitive = np.random.rand(self.dim)
                rand_social = np.random.rand(self.dim)
                particle_velocities[i] = (inertia_weight * particle_velocities[i]
                                          + self.alpha_cognitive * rand_cognitive * (personal_best_positions[i] - particle_positions[i])
                                          + self.alpha_social * rand_social * (global_best_position - particle_positions[i]))
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

                if np.random.rand() < self.mutation_prob:
                    mutation_step = np.random.normal(0, 0.1, self.dim)  # Increased mutation step standard deviation
                    particle_positions[i] += mutation_step
                    particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

                if np.random.rand() < self.crossover_prob:
                    partner_idx = np.random.choice(self.swarm_size)
                    rand_crossover = np.random.rand(self.dim) < 0.6  # Adjusted crossover threshold
                    particle_positions[i][rand_crossover] = personal_best_positions[partner_idx][rand_crossover]

                current_value = func(particle_positions[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = np.copy(particle_positions[i])

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = np.copy(particle_positions[i])

        return global_best_position, global_best_value