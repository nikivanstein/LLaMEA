import numpy as np

class HybridDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Increased population for better diversity
        self.inertia = 0.5  # Lower inertia to enhance convergence speed
        self.c1 = 1.5  # Adjusted cognitive component for individual search
        self.c2 = 1.5  # Balanced social component for coordinated exploration
        self.mutation_factor = 0.6  # Increased mutation for wider exploration
        self.crossover_rate = 0.8  # Slightly adjusted crossover rate for exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp_factor = 0.9  # More flexible velocity clamping

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.full(self.pop_size, float('inf'))
        global_best = particles[0].copy()
        global_best_fitness = float('inf')

        eval_count = 0
        iteration = 0
        while eval_count < self.budget:
            for i in range(self.pop_size):
                fitness = func(particles[i])
                eval_count += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()

                if eval_count >= self.budget:
                    break

            if eval_count >= self.budget:
                break

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocity_clamp = 2.0 * (1 - (iteration / (self.budget / self.pop_size)))  # Adaptive velocity clamp
            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best - particles))
            velocities = np.clip(velocities, -velocity_clamp, velocity_clamp)

            particles = particles + velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            for i in range(self.pop_size):
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant_vector = (personal_best[a] +
                                     self.mutation_factor * (personal_best[b] - personal_best[c]))
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate,
                                            mutant_vector, particles[i])

                    trial_fitness = func(trial_vector)
                    eval_count += 1

                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_fitness[i] = trial_fitness
                        personal_best[i] = trial_vector.copy()

                    if trial_fitness < global_best_fitness:
                        global_best_fitness = trial_fitness
                        global_best = trial_vector.copy()

                    if eval_count >= self.budget:
                        break
            iteration += 1

        return global_best