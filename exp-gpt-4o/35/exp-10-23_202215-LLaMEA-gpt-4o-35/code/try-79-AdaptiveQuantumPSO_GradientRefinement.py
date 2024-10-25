import numpy as np

class AdaptiveQuantumPSO_GradientRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 40  # Slightly adjusted population size for balance
        self.inertia_weight = 0.6  # Modified inertia weight for effective control
        self.cognitive_coefficient = 1.7  # Increased cognitive component for better personal search
        self.social_coefficient = 1.3  # Adjusted social component for exploration
        self.initial_temp = 1.0  # Reset initial temperature for controlled exploration
        self.cooling_rate = 0.85  # Modified cooling rate for improved convergence

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-2.0, 2.0, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired velocity update
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.normal(size=self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.normal(size=self.dim) * (personal_best[global_best_idx] - population[i]))

                # Gradient-based refinement
                refinement = np.sign(velocities[i]) * np.abs(velocities[i]**2)
                population[i] = np.clip(population[i] + refinement, self.bounds[0], self.bounds[1])

                # Evaluate new position
                fitness = func(population[i])
                self.evaluations += 1

                # Update personal best with adaptive acceptance
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
            self.inertia_weight = 0.4 + 0.3 * (self.budget - self.evaluations) / self.budget  

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]