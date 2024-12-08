import numpy as np

class EnhancedHybridOptV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Adjusted for efficiency
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.learning_rate = 0.05  # Adjusted learning rate
        self.inertia_weight = 0.6  # Modified inertia for better control
        self.cognitive_const = 1.2  # Reduced cognitive component
        self.social_const = 1.7  # Increased social component
        self.memory_factor = 0.5  # New memory factor for adaptive memory

    def dynamic_control_update(self, velocities, memory):
        dynamic_factor = np.sin(np.pi * self.eval_count / self.budget)
        return velocities * (1 - self.memory_factor) + dynamic_factor * memory

    def adaptive_memory_update(self, population, velocities, personal_best_positions, personal_best_scores, fitness):
        new_population = np.copy(population)
        memory = np.copy(velocities)

        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            
            # Update velocities with dynamic control
            velocities[i] = self.dynamic_control_update(velocities[i], memory[i])
            
            # Update velocities with Particle Swarm-like behavior
            velocities[i] += (
                self.cognitive_const * r1 * (personal_best_positions[i] - population[i]) +
                self.social_const * r2 * (self.best_solution - population[i])
            )
            
            # Update positions
            new_population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)
            new_fitness = fitness(new_population[i])
            self.eval_count += 1

            # Update personal bests
            if new_fitness < personal_best_scores[i]:
                personal_best_positions[i] = new_population[i]
                personal_best_scores[i] = new_fitness

            # Update global best
            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_population[i]

        return new_population, velocities, personal_best_positions, personal_best_scores

    def refined_gradient_update(self, solution, fitness):
        gradient_direction = np.random.uniform(-1.0, 1.0, self.dim)
        gradient_step_size = self.learning_rate * (1 - np.cos(self.eval_count * np.pi / (2 * self.budget)))
        adjusted_solution = np.clip(solution + gradient_step_size * gradient_direction, self.lower_bound, self.upper_bound)
        adjusted_fitness = fitness(adjusted_solution)
        self.eval_count += 1

        if adjusted_fitness < self.best_fitness:
            self.best_fitness = adjusted_fitness
            self.best_solution = adjusted_solution

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, float('inf'))

        # Evaluate initial population
        for i in range(self.population_size):
            fitness_value = func(population[i])
            self.eval_count += 1

            if fitness_value < self.best_fitness:
                self.best_fitness = fitness_value
                self.best_solution = population[i]

            personal_best_positions[i] = population[i]
            personal_best_scores[i] = fitness_value

        while self.eval_count < self.budget:
            # Apply Adaptive Memory Update
            population, velocities, personal_best_positions, personal_best_scores = self.adaptive_memory_update(
                population, velocities, personal_best_positions, personal_best_scores, func
            )

            # Apply Refined Gradient Update on the best solution found so far
            self.refined_gradient_update(self.best_solution, func)

        return self.best_solution