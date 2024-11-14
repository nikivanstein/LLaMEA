import numpy as np

class EnhancedAdaptiveMutationGADEWithDynamicPopulationSize:
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
        step_size_factor = 0.01

        def target_to_bounds(target, lower, upper):
            return np.clip(target, lower, upper)

        def local_search(candidate, f_local, eval_count):
            best_candidate = np.copy(candidate)
            step_size = step_size_factor
            for _ in range(max_ls_iter):
                new_candidate = target_to_bounds(best_candidate + step_size * np.random.randn(self.dim), -5.0, 5.0)
                eval_count += 1
                if f_local(new_candidate) < f_local(best_candidate):
                    best_candidate = new_candidate
                    step_size *= 1.2
                else:
                    step_size *= 0.8
            return best_candidate, eval_count

        eval_count = 0
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

                population[i], eval_count = local_search(population[i], func, eval_count)

                if func(population[i]) < best_fitness:
                    best_solution = population[i]
                    best_fitness = func(population[i])
                best_particle = global_best_pso(population, best_particle)

        return best_solution