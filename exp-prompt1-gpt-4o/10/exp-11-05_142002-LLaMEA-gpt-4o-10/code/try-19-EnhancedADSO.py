import numpy as np

class EnhancedADSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5, 2 * self.dim)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.iteration = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in personal_best_positions])
        
        # Update global best
        min_index = np.argmin(personal_best_values)
        self.global_best_position = personal_best_positions[min_index]
        self.global_best_value = personal_best_values[min_index]

        evaluations = self.population_size

        # Chaotic sequence for enhanced exploration
        chaotic_sequence = np.sin(np.linspace(0, 4 * np.pi, self.budget))

        # Main loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive Parameters with chaotic influence
                w = 0.5 + 0.5 * chaotic_sequence[self.iteration % len(chaotic_sequence)]
                c1 = 1.5 + chaotic_sequence[(self.iteration + 1) % len(chaotic_sequence)]
                c2 = 1.5 + chaotic_sequence[(self.iteration + 2) % len(chaotic_sequence)]

                # Update velocities and positions
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                                 c2 * np.random.rand(self.dim) * (self.global_best_position - population[i]))
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])
                
                # Evaluate new solution
                current_value = func(population[i])
                evaluations += 1

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_values[i] = current_value

                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_position = population[i]
                    self.global_best_value = current_value

                # Stop if budget is reached
                if evaluations >= self.budget:
                    break

            self.iteration += 1

        return self.global_best_position, self.global_best_value