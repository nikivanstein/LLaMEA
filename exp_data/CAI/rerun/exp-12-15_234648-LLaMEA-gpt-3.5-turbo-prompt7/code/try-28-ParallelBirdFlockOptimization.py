from concurrent.futures import ThreadPoolExecutor

class ParallelBirdFlockOptimization(BirdFlockOptimization):
    def __call__(self, func):
        def evaluate_population(population, res):
            for idx, ind in enumerate(population):
                res[idx] = func(ind)

        def parallel_fitness(population):
            fitness_values = np.zeros(len(population))
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(evaluate_population, ind, res) for ind, res in zip(population, fitness_values)]
                for future in futures:
                    future.result()
            return fitness_values

        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin(parallel_fitness(personal_best_pos))]

        for itr in range(self.budget):
            fitness_vals = parallel_fitness(population)
            for i in range(self.num_birds):
                velocity[i] = update_velocity(velocity[i], population[i], global_best_pos, personal_best_pos[i], itr)
                population[i] += velocity[i]
                if fitness_vals[i] < fitness(personal_best_pos[i]):
                    personal_best_pos[i] = population[i]
                if fitness(personal_best_pos[i]) < fitness(global_best_pos):
                    global_best_pos = personal_best_pos[i]

        return global_best_pos