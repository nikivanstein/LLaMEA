import numpy as np

class DynamicMultiSwarmPSO_ALF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Increased population for better exploration
        self.inertia = 0.5  # Base inertia for energy conservation
        self.c1 = 1.8  # Slightly higher cognitive component for stronger personal learning
        self.c2 = 1.5  # Balanced social component
        self.mutation_factor = 0.8  # Slightly modified mutation factor
        self.crossover_rate = 0.9  # Higher crossover rate for better diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = 4.0  # Increased clamping range
        self.num_swarms = 3  # Number of sub-swarms

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_swarms, self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_swarms, self.pop_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.full((self.num_swarms, self.pop_size), float('inf'))
        global_best = np.full((self.num_swarms, self.dim), float('inf'))
        global_best_fitness = np.full(self.num_swarms, float('inf'))

        eval_count = 0

        while eval_count < self.budget:
            # Evaluate fitness
            for s in range(self.num_swarms):
                for i in range(self.pop_size):
                    fitness = func(particles[s, i])
                    eval_count += 1

                    if fitness < personal_best_fitness[s, i]:
                        personal_best_fitness[s, i] = fitness
                        personal_best[s, i] = particles[s, i].copy()

                    if fitness < global_best_fitness[s]:
                        global_best_fitness[s] = fitness
                        global_best[s] = particles[s, i].copy()

                    if eval_count >= self.budget:
                        break

                if eval_count >= self.budget:
                    break

            # Update velocities and positions
            for s in range(self.num_swarms):
                r1 = np.random.rand(self.pop_size, self.dim)
                r2 = np.random.rand(self.pop_size, self.dim)

                # Dynamic adjustment of inertia using a non-linear formula
                self.inertia = 0.4 + 0.3 * np.exp(-3 * eval_count / self.budget)

                velocities[s] = (self.inertia * velocities[s] +
                                 self.c1 * r1 * (personal_best[s] - particles[s]) +
                                 self.c2 * r2 * (global_best[s] - particles[s]))
                velocities[s] = np.clip(velocities[s], -self.velocity_clamp, self.velocity_clamp)

                particles[s] += velocities[s]
                particles[s] = np.clip(particles[s], self.lower_bound, self.upper_bound)

                # Apply Adaptive Differential Mutation with Lévy Flights
                for i in range(self.pop_size):
                    if np.random.rand() < self.crossover_rate:
                        idxs = [idx for idx in range(self.pop_size) if idx != i]
                        a, b, c = np.random.choice(idxs, 3, replace=False)
                        alpha = 0.01
                        u = np.random.normal(0, 1, self.dim)
                        v = np.random.normal(0, 1, self.dim)
                        levy_flight = alpha * u / (np.abs(v) ** (1 / 3))

                        mutant_vector = (particles[s, a] +
                                         self.mutation_factor * (particles[s, b] - particles[s, c]) + levy_flight)
                        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                        trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate,
                                                mutant_vector, particles[s, i])

                        trial_fitness = func(trial_vector)
                        eval_count += 1

                        if trial_fitness < personal_best_fitness[s, i]:
                            personal_best_fitness[s, i] = trial_fitness
                            personal_best[s, i] = trial_vector.copy()

                        if trial_fitness < global_best_fitness[s]:
                            global_best_fitness[s] = trial_fitness
                            global_best[s] = trial_vector.copy()

                        if eval_count >= self.budget:
                            break

        best_swarm = np.argmin(global_best_fitness)
        return global_best[best_swarm]