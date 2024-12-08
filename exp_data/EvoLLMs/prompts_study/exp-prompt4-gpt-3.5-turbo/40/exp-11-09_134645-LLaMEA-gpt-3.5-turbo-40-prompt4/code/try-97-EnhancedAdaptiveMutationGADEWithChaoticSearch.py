import chaospy as cp

class EnhancedAdaptiveMutationGADEWithChaoticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_pop_size = 30
        max_ls_iter = 5
        omega = 0.5
        c1 = 1.5
        c2 = 1.5
        F_de = 0.8
        CR_de = 0.9

        def chaos_search(candidate):
            distribution = cp.Uniform(-0.1, 0.1)
            perturbation = distribution.sample(self.dim)
            return target_to_bounds(candidate + perturbation, -5.0, 5.0)

        def global_best_pso(particles, best_particle):
            return best_particle

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget // initial_pop_size):
            pop_size = initial_pop_size + int(10 * np.sin(0.1 * _))
            population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            velocities = np.zeros((pop_size, self.dim))
            best_particle = np.copy(best_solution)
            fitness_array = np.zeros(pop_size)
            for i in range(pop_size):
                parent1 = population[i]
                parent2 = population[np.random.choice(pop_size)]
                diversity = np.std(population)
                population[i] = crossover_ga(parent1, parent2, diversity)

                trial_de = mutate_de(population[i], population, F_de, CR_de, fitness_array)
                mutated_fitness_de = func(trial_de)
                fitness_array[i] = mutated_fitness_de
                if mutated_fitness_de < func(population[i]):
                    population[i] = trial_de

                velocities[i] = omega * velocities[i] + c1 * np.random.rand(self.dim) * (best_particle - population[i]) + c2 * np.random.rand(self.dim) * (best_solution - population[i])
                population[i] = target_to_bounds(population[i] + velocities[i], -5.0, 5.0)

                population = differential_evolution(population, func)

                population[i] = local_search(population[i], func)

                if func(population[i]) < best_fitness:
                    best_solution = population[i]
                    best_fitness = func(population[i])
                best_particle = global_best_pso(population, best_particle)

        return best_solution