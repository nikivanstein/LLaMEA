class ImprovedBirdFlockOptimization(BirdFlockOptimization):
    def __call__(self, func):
        inertia_weight = lambda t: 0.9 - t * (0.8 / self.budget)
        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin([fitness(ind) for ind in personal_best_pos])

        for t in range(self.budget):
            w = inertia_weight(t)
            for i in range(self.num_birds):
                velocity[i] = update_velocity(velocity[i], population[i], global_best_pos, personal_best_pos[i])
                population[i] += velocity[i]
                if fitness(population[i]) < fitness(personal_best_pos[i]):
                    personal_best_pos[i] = population[i]
                if fitness(personal_best_pos[i]) < fitness(global_best_pos):
                    global_best_pos = personal_best_pos[i]

        return global_best_pos