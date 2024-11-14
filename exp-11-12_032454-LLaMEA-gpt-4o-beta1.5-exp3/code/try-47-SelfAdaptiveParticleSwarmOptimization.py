import numpy as np

class SelfAdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.w_max = 0.9  # initial inertia weight
        self.w_min = 0.4  # final inertia weight
        self.c1 = 2.0  # personal learning factor
        self.c2 = 2.0  # global learning factor
        self.chaos_factor = 0.7  # factor for chaotic sequence influence

    def chaotic_sequence(self, size):
        x = np.random.rand()
        sequence = []
        for _ in range(size):
            x = 4 * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(ind) for ind in positions])
        num_evaluations = self.population_size

        # Determine global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        chaotic_seq = self.chaotic_sequence(self.budget)

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break

                # Update inertia weight dynamically
                w = self.w_max - ((self.w_max - self.w_min) * (num_evaluations / self.budget))

                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))

                # Chaotic perturbation
                chaotic_term = self.chaos_factor * chaotic_seq[num_evaluations % len(chaotic_seq)]
                velocities[i] += chaotic_term * (np.random.rand(self.dim) - 0.5)

                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                # Evaluate new fitness
                current_fitness = func(positions[i])
                num_evaluations += 1

                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = current_fitness

                    # Update global best
                    if current_fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = current_fitness

        return global_best_position, global_best_fitness