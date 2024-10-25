import numpy as np

class EQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 80  # Slightly increased population size for better exploration
        self.c1 = 1.5  # Adjusted cognitive coefficient for better convergence
        self.c2 = 1.5  # Adjusted social coefficient for balanced exploration
        self.inertia_weight = 0.7  # Increased inertia to maintain velocity
        self.epsilon = 1e-8
        self.mutation_factor = 0.9  # Enhanced mutation factor for diversity
        self.crossover_rate_base = 0.7  # Higher base crossover rate
        self.crossover_rate_max = 0.95

    def __call__(self, func):
        np.random.seed(42)
        lower_bound, upper_bound = self.bounds
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            quantum_step_size = np.random.uniform(0, (upper_bound - lower_bound) * 0.05, (self.population_size, self.dim))  # Increased quantum step size
            quantum_population = population + quantum_step_size * np.random.normal(0, 1, (self.population_size, self.dim))

            quantum_values = np.array([func(ind) for ind in quantum_population])
            evaluations += self.population_size
            if evaluations > self.budget:
                break

            for i in range(self.population_size):
                if quantum_values[i] < personal_best_values[i]:
                    personal_best_values[i] = quantum_values[i]
                    personal_best[i] = quantum_population[i]
                    if quantum_values[i] < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[i]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                inertia_weight_dynamic = self.inertia_weight * np.exp(-evaluations / self.budget)
                velocities[i] = (inertia_weight_dynamic * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - population[i]) +
                                 self.c2 * r2 * (global_best - population[i]))
                population[i] += velocities[i]

            population = np.clip(population, lower_bound, upper_bound)

            for i in range(self.population_size):
                crossover_rate = self.crossover_rate_base + (self.crossover_rate_max - self.crossover_rate_base) * \
                                 np.sqrt(1 - (evaluations / self.budget))
                if np.random.rand() < crossover_rate:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    mutation_vector = (population[indices[0]] +
                                       self.mutation_factor * (population[indices[1]] - population[indices[2]]))
                    mutant = np.clip(mutation_vector, lower_bound, upper_bound)
                    if evaluations < self.budget and func(mutant) < personal_best_values[i]:
                        population[i] = mutant
                        personal_best[i] = mutant
                        evaluations += 1

            for i in range(self.population_size):
                if evaluations < self.budget:
                    acceptance_prob = 0.5 + 0.25 * np.sin(np.pi * evaluations / self.budget)  # Modified acceptance probability for better control
                    if np.random.rand() < acceptance_prob:
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