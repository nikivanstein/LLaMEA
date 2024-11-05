import numpy as np

class EnhancedSwarmGradientOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, 10 * dim)  # Dynamic population sizing
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.learning_rate = 0.1
        self.inertia_weight = 0.5  # Reduced inertia for quicker convergence
        self.cognitive_const = 2.0  # Adjusted cognitive component
        self.social_const = 2.0  # Adjusted social component
    
    def swarm_update(self, population, velocities, personal_best_positions, personal_best_scores, fitness):
        new_population = np.copy(population)
        new_velocities = np.copy(velocities)

        for i in range(len(population)):
            r1, r2 = np.random.rand(2)
            
            # Update velocities
            new_velocities[i] = (
                self.inertia_weight * velocities[i] +
                self.cognitive_const * r1 * (personal_best_positions[i] - population[i]) +
                self.social_const * r2 * (self.best_solution - population[i])
            )
            
            # Update positions
            new_population[i] = np.clip(population[i] + new_velocities[i], self.lower_bound, self.upper_bound)
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

        return new_population, new_velocities, personal_best_positions, personal_best_scores

    def adaptive_update(self, solution, fitness):
        gradient_direction = np.random.normal(0, 1, self.dim)  # Gaussian perturbations
        gradient_step_size = self.learning_rate * (1 - self.eval_count / self.budget)
        for _ in range(3):  # Multiple trial updates
            candidate_solution = np.clip(solution + gradient_step_size * gradient_direction, self.lower_bound, self.upper_bound)
            candidate_fitness = fitness(candidate_solution)
            self.eval_count += 1

            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = candidate_solution

    def __call__(self, func):
        # Initialize population and velocities
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(population_size, float('inf'))

        # Evaluate initial population
        for i in range(population_size):
            fitness_value = func(population[i])
            self.eval_count += 1

            if fitness_value < self.best_fitness:
                self.best_fitness = fitness_value
                self.best_solution = population[i]

            personal_best_positions[i] = population[i]
            personal_best_scores[i] = fitness_value

        while self.eval_count < self.budget:
            # Apply Swarm Update
            population, velocities, personal_best_positions, personal_best_scores = self.swarm_update(
                population, velocities, personal_best_positions, personal_best_scores, func
            )

            # Apply Adaptive Update on the best solution found so far
            self.adaptive_update(self.best_solution, func)

        return self.best_solution