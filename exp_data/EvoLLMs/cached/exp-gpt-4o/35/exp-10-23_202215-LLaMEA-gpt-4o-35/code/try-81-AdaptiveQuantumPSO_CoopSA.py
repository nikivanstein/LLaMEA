import numpy as np

class AdaptiveQuantumPSO_CoopSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Adjusted population size for diversity
        self.inertia_weight = 0.6  # Lower inertia weight for faster convergence
        self.cognitive_coefficient = 1.8  # Increased cognitive component for personal search emphasis
        self.social_coefficient = 1.4  # Increased social component for better collaboration
        self.initial_temp = 1.0  # Moderate temperature for exploration
        self.cooling_rate = 0.85  # Slower cooling rate for sustained exploration
        self.alpha = 0.02  # New parameter for cooperative adjustment

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))  # Adjusted velocity range
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired update for velocities
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i])
                                 + self.alpha * np.random.rand(self.dim) * np.mean(population - population[i], axis=0))
                
                # Update position with quantum certainty
                population[i] = np.clip(population[i] + np.sign(velocities[i]) * np.abs(np.sinh(velocities[i])), self.bounds[0], self.bounds[1])

                # Evaluate new position
                fitness = func(population[i])
                self.evaluations += 1
                
                # Update personal best with cooperative SA
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
            self.inertia_weight = 0.5 + 0.25 * (self.budget - self.evaluations) / self.budget  # Updated dynamic range

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]