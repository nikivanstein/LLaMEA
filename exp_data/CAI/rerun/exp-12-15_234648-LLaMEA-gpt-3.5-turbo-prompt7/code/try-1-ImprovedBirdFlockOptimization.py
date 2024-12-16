class ImprovedBirdFlockOptimization(BirdFlockOptimization):
    def __call__(self, func):
        inertia_min, inertia_max = 0.1, 0.9
        inertia = inertia_max

        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin([fitness(ind) for ind in personal_best_pos])

        for _ in range(self.budget):
            for i in range(self.num_birds):
                inertia = inertia_max - (_ / self.budget) * (inertia_max - inertia_min)
                velocity[i] = update_velocity(velocity[i], population[i], global_best_pos, personal_best_pos[i])
                population[i] += inertia * velocity[i]
                if fitness(population[i]) < fitness(personal_best_pos[i]):
                    personal_best_pos[i] = population[i]
                if fitness(personal_best_pos[i]) < fitness(global_best_pos):
                    global_best_pos = personal_best_pos[i]

        return global_best_pos