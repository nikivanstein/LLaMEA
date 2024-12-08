import numpy as np

class QuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 30  # Slightly increased swarm size
        self.alpha_cognitive = 0.6  # Adjusted cognitive factor for better learning
        self.alpha_social = 0.4  # Boosted social factor
        self.inertia_start = 0.85  # Balanced starting inertia
        self.inertia_end = 0.15  # Slightly adjusted ending inertia
        self.mutation_prob = 0.25  # Further increased mutation probability
        self.crossover_prob = 0.55  # Further increased crossover probability

    def quantum_update(self, position, global_best):
        # Maintain diversity by adding a small uniform perturbation
        perturbation = np.random.uniform(-0.1, 0.1, position.shape)
        return np.random.normal(position, np.abs(global_best - position)) + perturbation

    def __call__(self, func):
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        particle_velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particle_positions)
        personal_best_values = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_end + (self.inertia_start - self.inertia_end) * ((self.budget - evaluations) / self.budget)
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                rand_cognitive = np.random.rand(self.dim)
                rand_social = np.random.rand(self.dim)
                r1, r2 = np.random.rand(2)
                # Dynamic adjustment of cognitive and social components
                particle_velocities[i] = (inertia_weight * particle_velocities[i]
                                         + r1 * self.alpha_cognitive * (personal_best_positions[i] - particle_positions[i])
                                         + r2 * self.alpha_social * (global_best_position - particle_positions[i]))
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

                if np.random.rand() < self.mutation_prob:
                    particle_positions[i] = self.quantum_update(particle_positions[i], global_best_position)
                    particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

                if np.random.rand() < self.crossover_prob:
                    partner_idx = np.random.choice(self.swarm_size)
                    rand_crossover = np.random.rand(self.dim) < 0.5
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