import numpy as np

class QuantumMemeticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia = 0.7  # Dynamic inertia for adaptive exploration
        self.c1 = 1.7  # Enhanced cognitive component for improved local search
        self.c2 = 1.3  # Reduced social component for diversity control
        self.quantum_factor = 0.5  # Quantum-inspired position update factor
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = 2.5  # Adjusted clamping for refined control
        self.eval_count = 0

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([float('inf')] * self.pop_size)
        global_best = particles[0].copy()
        global_best_fitness = float('inf')

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                fitness = func(particles[i])
                self.eval_count += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()

                if self.eval_count >= self.budget:
                    break

            if self.eval_count >= self.budget:
                break

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)

            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best - particles))
            velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)

            # Quantum-inspired update
            quantum_positions = particles + self.quantum_factor * np.random.randn(self.pop_size, self.dim)
            quantum_positions = np.clip(quantum_positions, self.lower_bound, self.upper_bound)

            particles = np.where(np.random.rand(self.pop_size, self.dim) < 0.5, quantum_positions, particles + velocities)
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Adaptive parameter adjustment
            self.inertia = 0.9 - (self.eval_count / self.budget) * 0.4

        return global_best