import numpy as np

class EnhancedADSOStarPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5, 2 * self.dim)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.iteration = 0
        self.stagnation_threshold = 10

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in personal_best_positions])
        min_index = np.argmin(personal_best_values)
        self.global_best_position = personal_best_positions[min_index]
        self.global_best_value = personal_best_values[min_index]
        evaluations = self.population_size
        chaotic_sequence = np.sin(np.linspace(0, 4 * np.pi, self.budget))
        stagnation_counter = 0
        mutation_rate = 0.05

        while evaluations < self.budget:
            prev_global_best_value = self.global_best_value
            for i in range(self.population_size):
                w = 0.5 + 0.5 * chaotic_sequence[self.iteration % len(chaotic_sequence)]
                c1 = 1.5 + chaotic_sequence[(self.iteration + 1) % len(chaotic_sequence)]
                c2 = 1.5 + chaotic_sequence[(self.iteration + 2) % len(chaotic_sequence)]

                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                                 c2 * np.random.rand(self.dim) * (self.global_best_position - population[i]))
                max_velocity = 0.1 * (self.bounds[1] - self.bounds[0])
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])
                
                # Introduce mutation-based perturbation
                if np.random.rand() < mutation_rate:
                    population[i] += np.random.normal(0, 0.1, self.dim)

                current_value = func(population[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_values[i] = current_value

                if current_value < self.global_best_value:
                    self.global_best_position = population[i]
                    self.global_best_value = current_value

                if evaluations >= self.budget:
                    break

            if self.global_best_value == prev_global_best_value:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= self.stagnation_threshold:
                for group in np.array_split(np.random.permutation(self.population_size), 2):
                    group_best_index = group[np.argmin(personal_best_values[group])]
                    for i in group:
                        if i != group_best_index:
                            population[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                stagnation_counter = 0

            self.iteration += 1

        return self.global_best_position, self.global_best_value