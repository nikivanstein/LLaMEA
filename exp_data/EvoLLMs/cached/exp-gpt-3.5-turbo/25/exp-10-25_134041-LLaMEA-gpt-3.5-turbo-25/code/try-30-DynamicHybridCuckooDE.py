# import numpy as np

class DynamicHybridCuckooDE(HybridCuckooDE):
    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def update_probability(success):
            self.pa = max(0.1, min(0.9, self.pa + 0.1 if success else -0.1))

        population = initialize_population()
        fitness_values = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness_values)
        best = population[best_idx].copy()

        for _ in range(self.max_iter):
            success = False
            for idx, ind in enumerate(population):
                if np.random.rand() < self.pa:
                    new_x = cuckoo_search_move(ind, best)
                    new_fitness = objective_function(new_x)
                    if new_fitness < fitness_values[idx]:
                        population[idx] = new_x
                        fitness_values[idx] = new_fitness
                        success = True
                else:
                    new_x = differential_evolution_move(population, idx, best)
                    new_fitness = objective_function(new_x)
                    if new_fitness < fitness_values[idx]:
                        population[idx] = new_x
                        fitness_values[idx] = new_fitness
                        success = True

            update_probability(success)

            best_idx = np.argmin(fitness_values)

            if fitness_values[best_idx] < fitness_values[best_idx]:
                best = population[best_idx]

        return best