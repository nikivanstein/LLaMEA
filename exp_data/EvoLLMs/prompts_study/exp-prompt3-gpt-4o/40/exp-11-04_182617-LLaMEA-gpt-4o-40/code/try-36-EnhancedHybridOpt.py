import numpy as np

class EnhancedHybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 15 * dim  # Increased population size
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.learning_rate = 0.05  # Reduced learning rate for finer updates
        self.inertia_weight = 0.6  # Adjusted inertia for improved convergence
        self.cognitive_const = 2.0  # Enhanced cognitive component
        self.social_const = 2.0  # Enhanced social component

    def particle_swarm_update(self, population, velocities, personal_best_positions, personal_best_scores, fitness):
        new_population = np.copy(population)
        new_velocities = np.copy(velocities)

        for i in range(self.population_size):
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

    def stochastic_perturbation(self):
        perturbation = np.random.normal(0, 0.1, self.dim)  # Gaussian perturbation
        perturbed_solution = np.clip(self.best_solution + perturbation, self.lower_bound, self.upper_bound)
        return perturbed_solution

    def adaptive_gradient_update(self, solution, fitness):
        gradient_direction = np.random.uniform(-1.0, 1.0, self.dim)
        gradient_step_size = self.learning_rate * (1 - self.eval_count / self.budget)
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
            # Apply Particle Swarm Update
            population, velocities, personal_best_positions, personal_best_scores = self.particle_swarm_update(
                population, velocities, personal_best_positions, personal_best_scores, func
            )

            # Use stochastic perturbation to explore the neighborhood of the best solution
            perturbed_solution = self.stochastic_perturbation()
            perturbed_fitness = func(perturbed_solution)
            self.eval_count += 1

            if perturbed_fitness < self.best_fitness:
                self.best_fitness = perturbed_fitness
                self.best_solution = perturbed_solution

            # Apply Adaptive Gradient Update on the best solution found so far
            self.adaptive_gradient_update(self.best_solution, func)

        return self.best_solution