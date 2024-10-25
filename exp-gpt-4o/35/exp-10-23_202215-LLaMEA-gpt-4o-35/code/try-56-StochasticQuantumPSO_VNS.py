import numpy as np

class StochasticQuantumPSO_VNS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 50  # Increased population size for more search agents
        self.inertia_weight = 0.6  # Refined inertia weight for a balanced exploration-exploitation ratio
        self.cognitive_coefficient = 1.4  # Reduced cognitive component for broader exploration
        self.social_coefficient = 1.3  # Increased social component for faster convergence
        self.initial_temp = 1.5  # Higher initial temperature for enhanced initial exploration
        self.cooling_rate = 0.85  # Slower cooling rate for prolonged annealing
        self.vns_radius = 2.0  # Radius for variable neighborhood search

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
                # Quantum-inspired update for velocities
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best[i] - population[i])
                                 + self.social_coefficient * np.random.rand(self.dim) * (personal_best[global_best_idx] - population[i]))
                
                # Update position with stochastic element
                stochastic_move = np.random.uniform(-1, 1, self.dim)
                population[i] = np.clip(population[i] + np.sign(velocities[i]) * np.abs(np.tanh(velocities[i])) + 0.1 * stochastic_move, self.bounds[0], self.bounds[1])

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

                # Variable Neighborhood Search
                neighborhood = np.random.uniform(-self.vns_radius, self.vns_radius, self.dim)
                neighbor_position = np.clip(population[i] + neighborhood, self.bounds[0], self.bounds[1])
                neighbor_fitness = func(neighbor_position)
                self.evaluations += 1
                if neighbor_fitness < personal_best_fitness[i]:
                    personal_best[i] = neighbor_position
                    personal_best_fitness[i] = neighbor_fitness

            # Update global best
            global_best_idx = np.argmin(personal_best_fitness)

            # Dynamic adjustment of inertia weight
            self.inertia_weight = 0.4 + 0.3 * (self.budget - self.evaluations) / self.budget  # Refined dynamic range

        return personal_best[global_best_idx], personal_best_fitness[global_best_idx]