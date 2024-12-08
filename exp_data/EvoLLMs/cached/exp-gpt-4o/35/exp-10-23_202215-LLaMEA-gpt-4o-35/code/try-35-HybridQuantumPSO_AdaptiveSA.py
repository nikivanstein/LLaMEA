import numpy as np

class HybridQuantumPSO_AdaptiveSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Increased population size for more diversity
        self.inertia_weight = 0.5  # Adjusted inertia weight for dynamic control
        self.cognitive_coefficient = 1.8  # Enhanced cognitive component for personal search
        self.social_coefficient = 1.2  # Modified social component for exploration
        self.initial_temp = 2.0  # Higher initial temperature for exploration
        self.cooling_rate = 0.9  # Slower cooling rate for temperature reduction

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-3.0, 3.0, (self.population_size, self.dim))  # Wider velocity range
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired nonlinear update for velocities
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.randn(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.randn(self.dim) * (personal_best[global_best_idx] - population[i]))
                
                # Update position with quantum certainty
                population[i] = np.clip(population[i] + np.sin(velocities[i]), self.bounds[0], self.bounds[1])

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

            # Dynamic adjustment of inertia weight
            self.inertia_weight = 0.3 + 0.4 * (self.budget - self.evaluations) / self.budget  # Expanded dynamic range

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]