import numpy as np

class EnhancedDynamicBoundaryHandlingFPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        self.explore_prob = 0.5  # Initial exploration probability
        self.mutation_rate = 0.5  # Initial mutation rate
        self.boundary_reflection_rate = 0.1  # Boundary reflection rate

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def dynamic_mutation(individual, best_pos, global_best_pos):
            mutation_strength = self.mutation_rate / (1 + np.linalg.norm(individual - global_best_pos))
            return individual + mutation_strength * np.random.normal(0, 1, size=self.dim)

        def constrain_boundary(position):
            return np.clip(position, -5.0, 5.0)

        def swarm_move(curr_pos, best_pos, global_best_pos, iteration):
            inertia_weight = 0.5 + 0.4 * (1 - iteration / self.max_iter)  # Dynamic inertia weight
            cognitive_weight = 1.5
            social_weight = 1.5
            velocity = np.zeros(self.dim)
            velocity = inertia_weight * velocity + cognitive_weight * np.random.rand() * (best_pos - curr_pos) + social_weight * np.random.rand() * (global_best_pos - curr_pos)
            return constrain_boundary(curr_pos + velocity)

        population = initialize_population()
        global_best_pos = population[np.argmin([func(ind) for ind in population])]

        for iteration in range(1, self.max_iter + 1):
            for i in range(self.population_size):
                if np.random.rand() < self.explore_prob:
                    new_pos = dynamic_mutation(population[i], global_best_pos, global_best_pos)
                    population[i] = new_pos if func(new_pos) < func(population[i]) else population[i]
                else:
                    new_pos = swarm_move(population[i], population[i], global_best_pos, iteration)
                    population[i] = new_pos if func(new_pos) < func(population[i]) else population[i]

                if func(population[i]) < func(global_best_pos):
                    global_best_pos = population[i]

                population[i] = constrain_boundary(population[i])

            self.mutation_rate *= 0.95  # Update mutation rate
            self.explore_prob = 0.5 * (1 - iteration / self.max_iter)  # Adapt exploration probability

        return global_best_pos