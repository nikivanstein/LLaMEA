import numpy as np

class HybridPSO_ADM_SCI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Population size
        self.inertia = 0.6  # Slightly increased inertia for exploration
        self.c1 = 1.4  # Slightly adjusted cognitive component
        self.c2 = 1.6  # Slightly adjusted social component
        self.mutation_factor = 0.9  # Enhanced mutation factor for more diversity
        self.crossover_rate = 0.85  # Slightly reduced crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = 2.0  # Initial velocity clamping

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.full(self.pop_size, float('inf'))
        global_best = particles[0].copy()
        global_best_fitness = float('inf')

        eval_count = 0

        while eval_count < self.budget:
            # Evaluate fitness
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

            # Update velocities and positions
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)

            # Adaptive velocity clamping
            self.velocity_clamp = 1.5 + 2.0 * (eval_count / self.budget)

            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best - particles))
            velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)

            particles = particles + velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Stochastic Component Integration - Enhanced Differential Mutation
            for i in range(self.pop_size):
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    
                    if np.random.rand() < 0.5:
                        mutant_vector = particles[a] + self.mutation_factor * (particles[b] - particles[c])
                    else:
                        mutant_vector = personal_best[a] + self.mutation_factor * (personal_best[b] - personal_best[c])
                    
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

        return global_best