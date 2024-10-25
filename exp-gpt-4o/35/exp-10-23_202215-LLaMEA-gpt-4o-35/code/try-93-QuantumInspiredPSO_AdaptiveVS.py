import numpy as np

class QuantumInspiredPSO_AdaptiveVS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Increased population size for greater diversity
        self.inertia_weight = 0.65  # Adjusted inertia weight for balanced velocity control
        self.cognitive_coefficient = 1.4  # Adjusted cognitive component for individual exploration
        self.social_coefficient = 1.3  # Enhanced social component for global convergence
        self.initial_temp = 1.0  # Lower initial temperature for refined convergence
        self.cooling_rate = 0.85  # Further adjusted cooling rate for optimal convergence

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1.5, 1.5, (self.population_size, self.dim))  # Dynamic velocity range
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired update for velocities with adaptive scaling
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i]))
                scale_factor = np.sqrt(np.sum(velocities[i]**2) / self.dim)
                velocities[i] /= (scale_factor + 1e-9)

                # Update position with quantum certainty
                population[i] = np.clip(population[i] + np.sign(velocities[i]) * np.abs(np.tanh(velocities[i])), self.bounds[0], self.bounds[1])

                # Evaluate new position
                fitness = func(population[i])
                self.evaluations += 1
                
                # Update personal best with enhanced SA
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

            # Dynamic adjustment of inertia weight
            self.inertia_weight = 0.6 + 0.15 * (self.budget - self.evaluations) / self.budget  # Modified dynamic range

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]