import numpy as np

class QG_CPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 50
        self.c1 = 1.7  # slightly increased cognitive coefficient
        self.c2 = 1.7  # slightly increased social coefficient
        self.inertia_weight = 0.7  # increased inertia for exploration
        self.epsilon = 1e-8
        self.mutation_factor = 0.85  # slightly increased mutation factor
        self.crossover_rate = 0.95  # further increased crossover rate

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

        def chaotic_sequence(length):
            # Generate a chaotic sequence using logistic map
            x = 0.7
            sequence = []
            for _ in range(length):
                x = 4 * x * (1 - x)
                sequence.append(x)
            return np.array(sequence)

        chaotic_weights = chaotic_sequence(self.budget)

        while evaluations < self.budget:
            # Chaotic-inspired random walks for inertia modulation
            chaotic_index = evaluations % len(chaotic_weights)
            inertia_weight = self.inertia_weight * chaotic_weights[chaotic_index]

            # Quantum-inspired random walks with enhanced variance
            quantum_step_size = np.random.uniform(-0.3, 0.3, self.dim)
            quantum_population = population + quantum_step_size * np.random.randn(self.population_size, self.dim)
            
            # Evaluate quantum-inspired population
            quantum_values = np.array([func(ind) for ind in quantum_population])
            evaluations += self.population_size
            if evaluations > self.budget:
                break

            # Update personal best with quantum-inspired population
            for i in range(self.population_size):
                if quantum_values[i] < personal_best_values[i]:
                    personal_best_values[i] = quantum_values[i]
                    personal_best[i] = quantum_population[i]
                    if quantum_values[i] < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[i]

            # Update velocities and positions with chaotic inertia
            r1, r2 = np.random.rand(), np.random.rand()
            velocities = (inertia_weight * velocities +
                          self.c1 * r1 * (personal_best - population) +
                          self.c2 * r2 * (global_best - population))
            population += velocities

            # Apply genetic-inspired mutation and crossover
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

            # Evaluate and update personal bests
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