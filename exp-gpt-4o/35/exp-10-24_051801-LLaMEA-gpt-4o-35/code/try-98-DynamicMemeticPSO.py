import numpy as np

class DynamicMemeticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 45  # Increased population for better diversity
        self.inertia = 0.5  # Reduced inertia to balance exploration-exploitation
        self.c1 = 1.5  # Adjusted cognitive component for improved exploration
        self.c2 = 1.5  # Balanced social component to avoid over-convergence
        self.mutation_factor = 0.65  # Enhanced mutation for broader search
        self.crossover_rate = 0.9  # Slightly increased crossover rate for variety
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_scale_factor = 0.7  # Adaptive scaling based on success rate

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.full(self.pop_size, float('inf'))
        global_best = particles[0].copy()
        global_best_fitness = float('inf')

        eval_count = 0
        iteration = 0
        success_rate = np.zeros(self.pop_size)

        while eval_count < self.budget:
            for i in range(self.pop_size):
                fitness = func(particles[i])
                eval_count += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()
                    success_rate[i] += 1  # Track success

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()

                if eval_count >= self.budget:
                    break

            if eval_count >= self.budget:
                break

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            success_probability = success_rate / (iteration + 1)
            velocity_scale = 1.0 + self.velocity_scale_factor * (success_probability - 0.5)
            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best - particles))
            velocities *= velocity_scale[:, np.newaxis]
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
                        success_rate[i] += 1

                    if trial_fitness < global_best_fitness:
                        global_best_fitness = trial_fitness
                        global_best = trial_vector.copy()

                    if eval_count >= self.budget:
                        break

            iteration += 1
            success_rate.fill(0)  # Reset success rate each iteration

        return global_best