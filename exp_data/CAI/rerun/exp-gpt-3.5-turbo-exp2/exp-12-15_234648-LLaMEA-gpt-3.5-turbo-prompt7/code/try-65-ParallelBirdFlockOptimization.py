class ParallelBirdFlockOptimization(BirdFlockOptimization):
    def __call__(self, func):
        def update_population(positions, velocities, personal_best_positions, global_best_position, iteration):
            new_positions = positions + velocities
            for i in range(self.num_birds):
                if fitness(new_positions[i]) < fitness(personal_best_positions[i]):
                    personal_best_positions[i] = new_positions[i]
                if fitness(personal_best_positions[i]) < fitness(global_best_position):
                    global_best_position = personal_best_positions[i]
            return new_positions, global_best_position

        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin([fitness(ind) for ind in personal_best_pos])]

        for itr in range(self.budget):
            new_positions, global_best_pos = update_population(population, velocity, personal_best_pos, global_best_pos, itr)
            population = new_positions

        return global_best_pos