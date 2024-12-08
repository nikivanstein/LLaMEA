import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 20  # Increased population size for diversity
        self.inertia_weight = 0.7  # Inertia weight for velocity update
        self.cognitive_coefficient = 1.4  # Cognitive coefficient
        self.social_coefficient = 1.4  # Social coefficient
        self.initial_temp = 1.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.98  # Cooling rate for simulated annealing

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocities and positions
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i]))
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                # Evaluate new position
                fitness = func(population[i])
                self.evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness

                # Simulated Annealing acceptance
                current_temp = self.initial_temp * (self.cooling_rate ** (self.evaluations / self.budget))
                if fitness >= personal_best_fitness[i]:
                    acceptance_prob = np.exp((personal_best_fitness[i] - fitness) / current_temp)
                    if np.random.rand() < acceptance_prob:
                        personal_best[i] = population[i]
                        personal_best_fitness[i] = fitness

            # Update global best
            global_best_idx = np.argmin(personal_best_fitness)

            # Adapt inertia weight
            self.inertia_weight = 0.4 + 0.3 * (self.budget - self.evaluations) / self.budget

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]