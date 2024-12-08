import numpy as np

class SwarmEnhancedGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.inertia_weight = 0.7
        self.social_coeff = 1.5
        self.cognitive_coeff = 1.5
        self.mutation_prob = 0.1
        self.eval_count = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def particle_movement(self, particle, velocity, personal_best, global_best):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_component = self.cognitive_coeff * r1 * (personal_best - particle)
        social_component = self.social_coeff * r2 * (global_best - particle)
        new_velocity = self.inertia_weight * velocity + cognitive_component + social_component
        new_particle = particle + new_velocity
        new_particle = np.clip(new_particle, self.lower_bound, self.upper_bound)
        return new_particle, new_velocity

    def mutate(self, individual):
        if np.random.rand() < self.mutation_prob:
            mutation_vector = np.random.normal(0, 1, self.dim)
            return np.clip(individual + mutation_vector, self.lower_bound, self.upper_bound)
        return individual

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)
        
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                # Particle movement inspired by PSO
                new_particle, new_velocity = self.particle_movement(population[i], velocities[i], personal_best[i], global_best)
                velocities[i] = new_velocity
                trial_fitness = func(new_particle)
                self.eval_count += 1

                # Check if the new particle is better
                if trial_fitness < fitness[i]:
                    population[i] = new_particle
                    fitness[i] = trial_fitness

                # Update personal best
                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = new_particle
                    personal_best_fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < global_best_fitness:
                    global_best = new_particle
                    global_best_fitness = trial_fitness

            # Apply mutation as a genetic algorithm operator
            for i in range(self.population_size):
                mutated = self.mutate(population[i])
                mutated_fitness = func(mutated)
                self.eval_count += 1
                if mutated_fitness < fitness[i]:
                    population[i] = mutated
                    fitness[i] = mutated_fitness

        return global_best