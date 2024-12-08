import numpy as np

class AdaptiveGeneticParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(6 * dim, 60)  # adaptive population size
        self.c1 = 1.494  # cognitive component
        self.c2 = 1.494  # social component
        self.inertia = 0.729  # inertia weight

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = population.copy()
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        evals = self.population_size

        while evals < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, self.lower_bound, self.upper_bound)

            # Evaluate fitness
            fitness = np.array([func(ind) for ind in population])
            evals += self.population_size

            # Update personal and global bests
            improved = fitness < personal_best_fitness
            personal_best_positions[improved] = population[improved]
            personal_best_fitness[improved] = fitness[improved]
            new_global_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[new_global_best_idx] < personal_best_fitness[global_best_idx]:
                global_best_idx = new_global_best_idx
                global_best_position = personal_best_positions[global_best_idx].copy()

            # Genetic algorithm inspired crossover and mutation
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                parent1, parent2 = population[np.random.choice(self.population_size, 2, replace=False)]
                crossover_point = np.random.randint(1, self.dim)
                child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                mutation = np.random.normal(0, 0.01, self.dim)
                child += mutation
                child = np.clip(child, self.lower_bound, self.upper_bound)
                child_fitness = func(child)
                evals += 1

                if child_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = child
                    personal_best_fitness[i] = child_fitness

        return global_best_position