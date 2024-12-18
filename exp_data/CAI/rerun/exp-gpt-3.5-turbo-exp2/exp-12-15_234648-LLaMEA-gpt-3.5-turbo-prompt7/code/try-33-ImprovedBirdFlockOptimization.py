import numpy as np
from joblib import Parallel, delayed

class ImprovedBirdFlockOptimization:
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))

        def fitness(position):
            return func(position)

        def parallel_fitness(positions):
            return Parallel(n_jobs=-1)(delayed(fitness)(pos) for pos in positions)

        def update_velocity(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget)  # Dynamic inertia weight
            return w * velocity + self.c1 * r1 * (personal_best_pos - position) + self.c2 * r2 * (global_best_pos - position)

        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin(parallel_fitness(personal_best_pos))]

        for itr in range(self.budget):
            for i in range(self.num_birds):
                velocity[i] = update_velocity(velocity[i], population[i], global_best_pos, personal_best_pos[i], itr)
                population[i] += velocity[i]
                if fitness(population[i]) < fitness(personal_best_pos[i]):
                    personal_best_pos[i] = population[i]
                if fitness(personal_best_pos[i]) < fitness(global_best_pos):
                    global_best_pos = personal_best_pos[i]

        return global_best_pos