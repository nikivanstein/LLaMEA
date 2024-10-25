import numpy as np

class HybridQuantumPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Increased population size for more diversity
        self.inertia_weight = 0.6  # Balanced inertia weight for better velocity control
        self.cognitive_coefficient = 1.8  # Enhanced cognitive component for stronger personal search
        self.social_coefficient = 1.0  # Slightly reduced social component for exploration
        self.mutation_coefficient = 0.8  # Differential mutation component
        self.initial_temp = 1.5  # Higher initial temperature for broader exploration
        self.cooling_rate = 0.85  # Adjusted cooling rate for improved convergence

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-2.5, 2.5, (self.population_size, self.dim))  # Broader velocity range
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
                
                # Apply differential mutation
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                mutant = personal_best[r1] + self.mutation_coefficient * (personal_best[r2] - personal_best[r3])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

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
            self.inertia_weight = 0.6 + 0.1 * (self.budget - self.evaluations) / self.budget  # Refined dynamic range

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]