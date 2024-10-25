import numpy as np

class HybridQuantumPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Increased population size for enhanced exploration
        self.inertia_weight = 0.6  # Moderate inertia weight for balanced velocity control
        self.cognitive_coefficient = 1.4  # Lower cognitive component for global focus
        self.social_coefficient = 1.3  # Slightly increased social component for improved cooperation
        self.initial_temp = 1.5  # Higher initial temperature for more robust exploration
        self.cooling_rate = 0.85  # Slower cooling rate to extend the search duration

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1.5, 1.5, (self.population_size, self.dim))  # Moderate velocity range
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive quantum-inspired velocity update
                adaptive_coeff = np.random.rand(self.dim) * (0.5 + (self.evaluations / self.budget))
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * adaptive_coeff * (personal_best[i] - population[i])
                                 + self.social_coefficient * adaptive_coeff * (global_best - population[i]))
                
                population[i] = np.clip(population[i] + np.sign(velocities[i]) * np.abs(np.tanh(velocities[i])), self.bounds[0], self.bounds[1])

                fitness = func(population[i])
                self.evaluations += 1
                
                # Update personal best with enhanced SA and diversity consideration
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
            global_best = personal_best[global_best_idx]

            # Dynamic inertia weight adjustment
            self.inertia_weight = 0.4 + 0.3 * (self.budget - self.evaluations) / self.budget

        return global_best, personal_best_fitness[global_best_idx]