import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 10 * dim
        self.inertia_weight = 0.7
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.quantum_param = 0.1

    def __call__(self, func):
        swarm = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.swarm_size, self.dim) * 0.1 * (self.upper_bound - self.lower_bound)

        fitness = np.apply_along_axis(func, 1, swarm)
        self.evaluations = self.swarm_size

        personal_best_positions = np.copy(swarm)
        personal_best_fitness = np.copy(fitness)

        global_best_idx = np.argmin(fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Quantum behavior
                quantum_positions = np.random.rand(self.dim) * self.quantum_param * (global_best_position - personal_best_positions[i])
                quantum_positions = np.clip(quantum_positions, self.lower_bound, self.upper_bound)

                # Update velocities
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.social_coeff * r2 * (global_best_position - swarm[i]))

                # Update positions
                swarm[i] = swarm[i] + velocities[i] + quantum_positions
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

                # Evaluate fitness
                fitness[i] = func(swarm[i])
                self.evaluations += 1

                # Update personal best
                if fitness[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness[i]
                    personal_best_positions[i] = swarm[i]

                    # Update global best
                    if fitness[i] < global_best_fitness:
                        global_best_fitness = fitness[i]
                        global_best_position = swarm[i]

        return global_best_position, global_best_fitness