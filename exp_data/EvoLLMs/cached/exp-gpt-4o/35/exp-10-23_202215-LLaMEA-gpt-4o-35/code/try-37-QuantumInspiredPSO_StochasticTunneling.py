import numpy as np

class QuantumInspiredPSO_StochasticTunneling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Increased population size for better diversity
        self.inertia_weight = 0.7  # Further dynamic adaptation for inertia weight
        self.cognitive_coefficient = 1.8  # Balanced cognitive component
        self.social_coefficient = 1.2  # Enhanced social component for collaboration
        self.initial_temp = 1.5  # Higher initial temperature for more exploration
        self.cooling_rate = 0.9  # Slower cooling rate for gradual temperature reduction

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-3.0, 3.0, (self.population_size, self.dim))  # Increased velocity range
        personal_best = population.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        global_best_idx = np.argmin(personal_best_fitness)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i]))
                
                population[i] = np.clip(population[i] + np.sign(velocities[i]) * np.abs(np.tanh(velocities[i])), self.bounds[0], self.bounds[1])

                fitness = func(population[i])
                self.evaluations += 1
                
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                else:
                    current_temp = self.initial_temp * (self.cooling_rate ** (self.evaluations / self.budget))
                    acceptance_prob = np.exp(-abs(personal_best_fitness[i] - fitness) / (current_temp + 1e-10))
                    if np.random.rand() < acceptance_prob:
                        personal_best[i] = population[i]
                        personal_best_fitness[i] = fitness

            global_best_idx = np.argmin(personal_best_fitness)
            self.inertia_weight = 0.5 + 0.3 * np.tanh((self.budget - self.evaluations) / self.budget)

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]