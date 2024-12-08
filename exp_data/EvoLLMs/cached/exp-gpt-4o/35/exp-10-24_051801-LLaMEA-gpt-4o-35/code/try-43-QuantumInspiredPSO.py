import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 60  # Increased population size for better diversity
        self.inertia = 0.4  # Lower inertia to enhance convergence speed
        self.c1 = 1.7  # Slightly higher cognitive component for individual learning
        self.c2 = 1.3  # Slightly lower social component to reduce herd behavior
        self.mutation_factor = 0.6  # Increased mutation factor for higher adaptability
        self.crossover_rate = 0.8  # Lower crossover rate to stabilize exploitation
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = 2.0  # Reduced clamping for finer adjustments
        self.quantum_prob = 0.2  # Probability of applying quantum perturbation

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([float('inf')] * self.pop_size)
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

            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best - particles))
            velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)

            particles = particles + velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Apply Hybrid Search with Quantum Perturbation
            for i in range(self.pop_size):
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant_vector = (personal_best[a] +
                                     self.mutation_factor * (personal_best[b] - personal_best[c]))
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate,
                                            mutant_vector, particles[i])

                    if np.random.rand() < self.quantum_prob:
                        quantum_vector = global_best + np.random.normal(0, 1, self.dim) * 0.1
                        quantum_vector = np.clip(quantum_vector, self.lower_bound, self.upper_bound)
                        trial_vector = np.where(np.random.rand(self.dim) < self.quantum_prob,
                                                quantum_vector, trial_vector)

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