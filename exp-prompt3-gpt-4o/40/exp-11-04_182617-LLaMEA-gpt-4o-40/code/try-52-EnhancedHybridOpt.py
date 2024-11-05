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
        self.learning_rate = 0.1
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.temperature = 1.0  # Simulated annealing component

    def particle_swarm_update(self, population, velocities, personal_best_positions, personal_best_scores, fitness):
        new_population = np.copy(population)
        new_velocities = np.copy(velocities)

        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            
            new_velocities[i] = (
                self.inertia_weight * velocities[i] +
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

    def simulated_annealing_update(self, solution, fitness):
        perturbation = np.random.uniform(-1.0, 1.0, self.dim)
        step_size = self.learning_rate * (1 - self.eval_count / self.budget)
        proposal_solution = np.clip(solution + step_size * perturbation, self.lower_bound, self.upper_bound)
        proposal_fitness = fitness(proposal_solution)
        self.eval_count += 1

        if proposal_fitness < self.best_fitness or np.random.rand() < np.exp((self.best_fitness - proposal_fitness) / self.temperature):
            self.best_fitness = proposal_fitness
            self.best_solution = proposal_solution

        # Update temperature
        self.temperature *= 0.99

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
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

            self.simulated_annealing_update(self.best_solution, func)

        return self.best_solution