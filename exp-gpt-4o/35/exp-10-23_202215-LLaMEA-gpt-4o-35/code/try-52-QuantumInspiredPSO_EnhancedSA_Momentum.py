import numpy as np

class QuantumInspiredPSO_EnhancedSA_Momentum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Adjusted for increased exploration
        self.inertia_weight = 0.6  # Optimized inertia for balanced exploration
        self.cognitive_coefficient = 1.4  # Adjusted for improved local search
        self.social_coefficient = 1.3  # Enhanced social component for better group dynamics
        self.momentum = 0.9  # Momentum for velocity smoothing
        self.initial_temp = 1.0  # Optimized initial temperature
        self.cooling_rate = 0.85  # Faster cooling for quicker convergence

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-2.5, 2.5, (self.population_size, self.dim))  # Wider initial velocity
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                velocities[i] = (self.momentum * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i]))

                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                fitness = func(population[i])
                self.evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                else:
                    current_temp = self.initial_temp * (self.cooling_rate ** (self.evaluations / self.budget))
                    acceptance_prob = np.exp((personal_best_fitness[i] - fitness) / (current_temp + 1e-10))
                    if np.random.rand() < acceptance_prob:
                        personal_best[i] = population[i]
                        personal_best_fitness[i] = fitness

            global_best_idx = np.argmin(personal_best_fitness)

            self.inertia_weight = 0.4 + 0.3 * (self.budget - self.evaluations) / self.budget  # Smoother dynamic range

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]