import numpy as np

class HybridPSOASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(0.5 * dim)
        self.omega = 0.5  # Inertia weight
        self.phi_p = 0.5  # Cognitive parameter
        self.phi_g = 0.5  # Social parameter
        self.current_evaluations = 0
        self.temperature = 1.0

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.random.uniform(
            -1, 1, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break

                # Update velocity
                velocities[i] = (
                    self.omega * velocities[i]
                    + self.phi_p * np.random.rand(self.dim) * (personal_best[i] - population[i])
                    + self.phi_g * np.random.rand(self.dim) * (global_best - population[i])
                )
                # Update position
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                trial_fitness = func(population[i])
                self.current_evaluations += 1

                # Adaptive simulated annealing acceptance
                delta_e = trial_fitness - fitness[i]
                acceptance_probability = np.exp(-delta_e / (self.temperature + 1e-9))

                # Adaptive temperature schedule
                self.temperature *= 0.99

                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = population[i]
                        personal_best_fitness[i] = trial_fitness

                    if trial_fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = trial_fitness

        return global_best

# Example of instantiation and usage:

# optimizer = HybridPSOASA(budget=2000, dim=10)
# best_solution = optimizer(my_black_box_function)