import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedBirdFlockOptimization:
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5, num_threads=4):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_threads = num_threads

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))

        def fitness(position):
            return func(position)

        def update_velocity(i, velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget)  # Dynamic inertia weight
            velocity[i] = w * velocity[i] + self.c1 * r1 * (personal_best_pos[i] - position[i]) + self.c2 * r2 * (global_best_pos - position[i])
            position[i] += velocity[i]
            if fitness(position[i]) < fitness(personal_best_pos[i]):
                personal_best_pos[i] = position[i]
            return fitness(personal_best_pos[i]), position[i]

        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin([fitness(ind) for ind in personal_best_pos])]

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for itr in range(self.budget):
                futures = [executor.submit(update_velocity, i, velocity[i], population[i], global_best_pos, personal_best_pos, itr) for i in range(self.num_birds)]
                results = [future.result() for future in futures]
                for i, (fit_val, pos) in enumerate(results):
                    if fit_val < fitness(global_best_pos):
                        global_best_pos = pos

        return global_best_pos