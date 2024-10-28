import numpy as np

class HybridPSO_AdaptiveSA_DynamicPop:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.initial_population_size = 30  # Increased initial population size for more exploration
        self.inertia_weight = 0.7  # Tweaked inertia weight for balancing exploration and exploitation
        self.cognitive_coefficient = 1.7  # Modified cognitive coefficient
        self.social_coefficient = 1.5  # Modified social coefficient
        self.initial_temp = 0.9  # Slightly increased initial temperature for SA
        self.cooling_rate = 0.9  # Adjusted cooling rate for a more gradual cooling
        self.pop_shrink_rate = 0.98  # New parameter for dynamic population resizing

    def __call__(self, func):
        # Initialize population and velocities
        population_size = self.initial_population_size
        population = np.random.uniform(self.bounds[0], self.bounds[1], (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = population_size

        while self.evaluations < self.budget:
            for i in range(population_size):
                # Update velocities with tweaked weight
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i]))
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                # Evaluate new position
                fitness = func(population[i])
                self.evaluations += 1

                # Update personal best with adaptive SA
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                else:
                    current_temp = self.initial_temp * (self.cooling_rate ** (self.evaluations / self.budget))
                    acceptance_prob = np.exp((personal_best_fitness[i] - fitness) / (current_temp + 1e-10))
                    if np.random.rand() < acceptance_prob:
                        personal_best[i] = population[i]
                        personal_best_fitness[i] = fitness

            # Update global best
            global_best_idx = np.argmin(personal_best_fitness)

            # Adapt inertia weight dynamically
            self.inertia_weight = 0.4 + 0.3 * (self.budget - self.evaluations) / self.budget

            # Dynamic population resizing
            if self.evaluations < self.budget * 0.8:
                population_size = max(5, int(self.initial_population_size * (self.pop_shrink_rate ** (self.evaluations / self.budget))))

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]