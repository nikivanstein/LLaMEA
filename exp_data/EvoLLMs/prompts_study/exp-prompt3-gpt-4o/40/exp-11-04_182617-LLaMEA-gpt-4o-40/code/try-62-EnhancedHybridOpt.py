import numpy as np

class EnhancedHybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.initial_learning_rate = 0.1
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4  
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.learning_rate = self.initial_learning_rate

    def particle_swarm_update(self, population, velocities, personal_best_positions, personal_best_scores, fitness):
        new_population = np.copy(population)
        new_velocities = np.copy(velocities)
        inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (self.eval_count / self.budget)

        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            
            new_velocities[i] = (
                inertia_weight * velocities[i] +
                self.cognitive_const * r1 * (personal_best_positions[i] - population[i]) +
                self.social_const * r2 * (self.best_solution - population[i])
            )
            
            new_population[i] = np.clip(population[i] + new_velocities[i], self.lower_bound, self.upper_bound)
            new_fitness = fitness(new_population[i])
            self.eval_count += 1

            if new_fitness < personal_best_scores[i]:
                personal_best_positions[i] = new_population[i]
                personal_best_scores[i] = new_fitness

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_population[i]

        return new_population, new_velocities, personal_best_positions, personal_best_scores

    def adaptive_gradient_update(self, solution, fitness):
        gradient_direction = np.random.normal(0.0, 1.0, self.dim)
        gradient_step_size = self.learning_rate * (1 - self.eval_count / self.budget)
        adjusted_solution = np.clip(solution + gradient_step_size * gradient_direction, self.lower_bound, self.upper_bound)
        adjusted_fitness = fitness(adjusted_solution)
        self.eval_count += 1

        if adjusted_fitness < self.best_fitness:
            self.best_fitness = adjusted_fitness
            self.best_solution = adjusted_solution

        self.learning_rate *= 0.99  

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim)) * ((1 + np.random.normal(0, 0.1, (self.population_size, self.dim))))
        velocities = np.zeros((self.population_size, self.dim))
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
            population, velocities, personal_best_positions, personal_best_scores = self.particle_swarm_update(
                population, velocities, personal_best_positions, personal_best_scores, func
            )
            self.adaptive_gradient_update(self.best_solution, func)

        return self.best_solution