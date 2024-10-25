import numpy as np

class QuantumInspiredPSO_EnhancedSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Increased population size for improved diversity
        self.inertia_weight = 0.6  # Adjusted inertia weight for better exploration-exploitation balance
        self.cognitive_coefficient = 1.8  # Increased cognitive component for more personal search exploration
        self.social_coefficient = 1.5  # Increased social component for better global exploration
        self.initial_temp = 1.5  # Higher initial temperature for enhanced exploration
        self.cooling_rate = 0.85  # Slower cooling rate for prolonged convergence

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-3.0, 3.0, (self.population_size, self.dim))  # Broader velocity range
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired update for velocities
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i]))
                
                # Update position with improved quantum certainty
                population[i] = np.clip(population[i] + np.tanh(velocities[i]), self.bounds[0], self.bounds[1])

                # Evaluate new position
                fitness = func(population[i])
                self.evaluations += 1

                # Dual adaptive simulated annealing for personal best update
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

            # Dynamic adjustment of inertia weight for emphasis on exploitation
            self.inertia_weight = 0.4 + 0.3 * (self.budget - self.evaluations) / self.budget

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]