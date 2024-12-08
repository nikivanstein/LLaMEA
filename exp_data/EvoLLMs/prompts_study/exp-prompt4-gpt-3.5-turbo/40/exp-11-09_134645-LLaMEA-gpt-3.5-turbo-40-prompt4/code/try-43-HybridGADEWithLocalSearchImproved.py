import numpy as np

class HybridGADEWithLocalSearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        max_ls_iter = 5
        omega = 0.5
        c1 = 1.5
        c2 = 1.5
        F_min = 0.1
        F_max = 0.9
        CR_min = 0.1
        CR_max = 0.9

        def target_to_bounds(target, lower, upper):
            return np.clip(target, lower, upper)

        def local_search(candidate, f_local):
            best_candidate = np.copy(candidate)
            for _ in range(max_ls_iter):
                new_candidate = target_to_bounds(best_candidate + 0.01 * np.random.randn(self.dim), -5.0, 5.0)
                if f_local(new_candidate) < f_local(best_candidate):
                    best_candidate = new_candidate
            return best_candidate

        def crossover_ga(parent1, parent2):
            child = np.copy(parent1)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    child[i] = parent2[i]
            return child

        def mutate_de(current, candidates, F, CR):
            mutated = np.copy(current)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)
                    if np.random.rand() < 0.5:
                        F_i = F + 0.1 * np.random.randn()
                        F_i = target_to_bounds(F_i, F_min, F_max)  # Adaptive adjustment of F_i
                    else:
                        F_i = F
                    mutated[i] = target_to_bounds(candidates[a, i] + F_i * (candidates[b, i] - candidates[c, i]), -5.0, 5.0)
            return mutated

        def global_best_pso(particles, best_particle):
            return best_particle

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        F = 0.8
        CR = 0.9

        for _ in range(self.budget // pop_size):
            population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            velocities = np.zeros((pop_size, self.dim))
            best_particle = np.copy(best_solution)
            for i in range(pop_size):
                # GA Crossover
                parent1 = population[i]
                parent2 = population[np.random.choice(pop_size)]
                population[i] = crossover_ga(parent1, parent2)

                # DE Mutation
                trial_de = mutate_de(population[i], population, F, CR)
                mutated_fitness_de = func(trial_de)
                if mutated_fitness_de < func(population[i]):
                    population[i] = trial_de

                # PSO Update
                velocities[i] = omega * velocities[i] + c1 * np.random.rand(self.dim) * (best_particle - population[i]) + c2 * np.random.rand(self.dim) * (best_solution - population[i])
                population[i] = target_to_bounds(population[i] + velocities[i], -5.0, 5.0)

                # Local Search
                population[i] = local_search(population[i], func)

                # Update best solutions
                if func(population[i]) < best_fitness:
                    best_solution = population[i]
                    best_fitness = func(population[i])
                best_particle = global_best_pso(population, best_particle)

        return best_solution