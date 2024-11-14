import numpy as np

class HybridPSODEAlgorithm:
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

        def target_to_bounds(target, lower, upper):
            return np.clip(target, lower, upper)

        def local_search(candidate, f_local):
            best_candidate = np.copy(candidate)
            step_size = 0.01
            for _ in range(max_ls_iter):
                new_candidate = target_to_bounds(best_candidate + step_size * np.random.randn(self.dim), -5.0, 5.0)
                if f_local(new_candidate) < f_local(best_candidate):
                    best_candidate = new_candidate
                    step_size *= 1.2  
                else:
                    step_size *= 0.8  
            return best_candidate

        def crossover_ga(parent1, parent2, diversity):
            child = np.copy(parent1)
            prob_crossover = 0.5 + 0.4 * diversity  
            for i in range(self.dim):
                if np.random.rand() < prob_crossover:
                    child[i] = parent2[i]
            return child

        def mutate_de(current, candidates, F, CR, fitness_array):
            mutated = np.copy(current)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    a, b, c = np.random.choice(initial_pop_size, 3, replace=False)
                    F_i = F + 0.1 * np.random.randn()
                    F_i = max(0.1, min(0.9, F_i))
                    if fitness_array[a] > np.mean(fitness_array):
                        F_i *= 1.1
                    else:
                        F_i *= 0.9
                    mutated[i] = target_to_bounds(candidates[a, i] + F_i * (candidates[b, i] - candidates[c, i]), -5.0, 5.0)
            return mutated

        def global_best_pso(particles, best_particle):
            return best_particle

        def differential_evolution(population, func):
            new_population = np.copy(population)
            for i in range(len(population)):
                a, b, c = np.random.choice(len(population), 3, replace=False)
                F_i = F_de + 0.1 * np.random.randn()
                F_i = max(0.1, min(0.9, F_i)) 
                trial_de = target_to_bounds(population[a] + F_i * (population[b] - population[c]), -5.0, 5.0)
                if func(trial_de) < func(population[i]):
                    new_population[i] = trial_de
            return new_population

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