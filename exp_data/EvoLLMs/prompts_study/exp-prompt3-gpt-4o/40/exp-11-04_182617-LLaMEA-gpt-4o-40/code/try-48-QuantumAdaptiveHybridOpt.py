import numpy as np

class QuantumAdaptiveHybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Slightly larger for increased diversity
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.learning_rate = 0.2  # Enhanced learning rate
        self.quantum_gate_angle = 0.05  # Rotation angle for quantum-inspired update
        self.cognitive_const = 1.4  # Adjusted PSO cognitive component
        self.social_const = 1.6  # Adjusted PSO social component

    def quantum_rotation_update(self, population, personal_best_positions, personal_best_scores, fitness):
        new_population = np.copy(population)

        for i in range(self.population_size):
            rotation_matrix = np.array([[np.cos(self.quantum_gate_angle), -np.sin(self.quantum_gate_angle)],
                                        [np.sin(self.quantum_gate_angle), np.cos(self.quantum_gate_angle)]])
            rotation_vector = np.random.uniform(-1.0, 1.0, self.dim)
            rotated_vector = np.dot(rotation_matrix, rotation_vector)
            new_population[i] = np.clip(population[i] + rotated_vector, self.lower_bound, self.upper_bound)
            new_fitness = fitness(new_population[i])
            self.eval_count += 1

            if new_fitness < personal_best_scores[i]:
                personal_best_positions[i] = new_population[i]
                personal_best_scores[i] = new_fitness

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_population[i]

        return new_population, personal_best_positions, personal_best_scores

    def dynamic_gradient_update(self, solution, fitness):
        gradient_direction = np.random.uniform(-1.0, 1.0, self.dim)
        gradient_step_size = self.learning_rate * np.exp(-self.eval_count / self.budget)
        adjusted_solution = np.clip(solution + gradient_step_size * gradient_direction, self.lower_bound, self.upper_bound)
        adjusted_fitness = fitness(adjusted_solution)
        self.eval_count += 1

        if adjusted_fitness < self.best_fitness:
            self.best_fitness = adjusted_fitness
            self.best_solution = adjusted_solution

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, float('inf'))

        for i in range(self.population_size):
            fitness_value = func(population[i])
            self.eval_count += 1

            if fitness_value < self.best_fitness:
                self.best_fitness = fitness_value
                self.best_solution = population[i]

            personal_best_positions[i] = population[i]
            personal_best_scores[i] = fitness_value

        while self.eval_count < self.budget:
            population, personal_best_positions, personal_best_scores = self.quantum_rotation_update(
                population, personal_best_positions, personal_best_scores, func
            )
            
            self.dynamic_gradient_update(self.best_solution, func)

        return self.best_solution