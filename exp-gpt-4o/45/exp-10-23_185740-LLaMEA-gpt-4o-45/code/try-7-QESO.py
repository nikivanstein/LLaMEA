import numpy as np

class QESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 60  # Adjusted for better exploration
        self.c1 = 1.5  # Improved cognitive coefficient
        self.c2 = 1.5  # Improved social coefficient
        self.inertia_weight = 0.7
        self.epsilon = 1e-6  # Enhanced precision
        self.mutation_factor = 0.9  # Enhanced mutation factor
        self.crossover_rate = 0.85
        self.quantum_factor = 0.03  # Quantum influence factor

    def __call__(self, func):
        np.random.seed(42)
        lower_bound, upper_bound = self.bounds
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Quantum-inspired position update
            quantum_shift = np.random.uniform(-self.quantum_factor, self.quantum_factor, (self.population_size, self.dim))
            population += quantum_shift

            # Compute gradients for a subset of particles
            gradient = np.zeros_like(population)
            chosen_indices = np.random.choice(self.population_size, self.population_size // 3, replace=False)
            for i in chosen_indices:
                original_value = func(population[i])
                for d in range(self.dim):
                    step = np.zeros(self.dim)
                    step[d] = self.epsilon
                    gradient[i][d] = (func(population[i] + step) - original_value) / self.epsilon
                    evaluations += 1
                    if evaluations >= self.budget:
                        break
                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            r1, r2 = np.random.rand(), np.random.rand()
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best - population) +
                          self.c2 * r2 * (global_best - population) -
                          0.02 * gradient)  # Enhanced gradient influence
            population += velocities

            # Apply differential evolution strategy
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    mutation_vector = (population[indices[0]] +
                                       self.mutation_factor * (population[indices[1]] - population[indices[2]]))
                    mutant = np.clip(mutation_vector, lower_bound, upper_bound)
                    if func(mutant) < personal_best_values[i]:
                        population[i] = mutant
                        personal_best[i] = mutant

            # Enforce bounds
            population = np.clip(population, lower_bound, upper_bound)

            # Evaluate and update personal and global bests
            for i in range(self.population_size):
                if evaluations < self.budget:
                    fitness = func(population[i])
                    evaluations += 1
                    if fitness < personal_best_values[i]:
                        personal_best_values[i] = fitness
                        personal_best[i] = population[i]
                    if fitness < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[i]
                else:
                    break

        return global_best