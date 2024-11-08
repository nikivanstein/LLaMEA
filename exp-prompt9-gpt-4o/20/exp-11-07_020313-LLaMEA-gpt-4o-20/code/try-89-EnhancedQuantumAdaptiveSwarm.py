import numpy as np

class EnhancedQuantumAdaptiveSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 40  # Increased swarm size for better exploration
        self.alpha_cognitive = 0.6  # Increased cognitive factor for enhanced personal learning
        self.alpha_social = 0.3  # Reduced social factor for diversified convergence
        self.inertia_start = 0.85  # Slightly reduced initial inertia for balanced exploration
        self.inertia_end = 0.3  # Increased final inertia for smoother transition
        self.mutation_prob = 0.25  # Increased mutation probability for diversity
        self.crossover_prob = 0.7  # Higher crossover rate for effective diversity

    def quantum_update(self, position, global_best, iteration, max_iterations):
        # Adjusted quantum-inspired update with adaptable exploration-exploitation balance
        sigma = np.abs(global_best - position) * np.exp(-iteration / max_iterations)
        return np.random.normal(position, sigma)

    def __call__(self, func):
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        particle_velocities = np.random.uniform(-0.3, 0.3, (self.swarm_size, self.dim))  # Narrowed velocity range for stability
        personal_best_positions = np.copy(particle_positions)
        personal_best_values = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.swarm_size
        iteration = 0
        max_iterations = self.budget // self.swarm_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_end + (self.inertia_start - self.inertia_end) * ((self.budget - evaluations) / self.budget)
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
                    particle_positions[i] = self.quantum_update(particle_positions[i], global_best_position, iteration, max_iterations)
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

            iteration += 1

        return global_best_position, global_best_value