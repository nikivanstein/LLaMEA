import numpy as np

class ImprovedBirdFlockOptimization(BirdFlockOptimization):
    def __call__(self, func):
        def update_velocity_multi_strategy(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget)  # Dynamic inertia weight
            strategy = np.random.randint(3)  # Randomly select update strategy
            if strategy == 0:
                return w * velocity + self.c1 * r1 * (personal_best_pos - position) + self.c2 * r2 * (global_best_pos - position)
            elif strategy == 1:
                return w * velocity + self.c1 * r1 * (personal_best_pos - position)
            else:
                return w * velocity + self.c2 * r2 * (global_best_pos - position)

        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin([fitness(ind) for ind in personal_best_pos])]

        for itr in range(self.budget):
            for i in range(self.num_birds):
                velocity[i] = update_velocity_multi_strategy(velocity[i], population[i], global_best_pos, personal_best_pos[i], itr)
                population[i] += velocity[i]
                if fitness(population[i]) < fitness(personal_best_pos[i]):
                    personal_best_pos[i] = population[i]
                if fitness(personal_best_pos[i]) < fitness(global_best_pos):
                    global_best_pos = personal_best_pos[i]

        return global_best_pos