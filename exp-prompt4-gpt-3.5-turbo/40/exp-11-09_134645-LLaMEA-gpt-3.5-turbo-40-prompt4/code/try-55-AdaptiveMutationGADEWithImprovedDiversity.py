import numpy as np

class AdaptiveMutationGADEWithImprovedDiversity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        max_ls_iter = 5
        omega = 0.5
        c1 = 1.5
        c2 = 1.5

        def target_to_bounds(target, lower, upper):
            return np.clip(target, lower, upper)

        def local_search(candidate, f_local):
            best_candidate = np.copy(candidate)
            for _ in range(max_ls_iter):
                new_candidate = target_to_bounds(best_candidate + 0.01 * np.random.randn(self.dim), -5.0, 5.0)
                if f_local(new_candidate) < f_local(best_candidate):
                    best_candidate = new_candidate
            return best_candidate

        def crossover_ga(parent1, parent2, diversity):
            child = np.copy(parent1)
            prob_crossover = 0.5 + 0.4 * diversity  # Dynamic adaptation of crossover probability
            for i in range(self.dim):
                if np.random.rand() < prob_crossover:
                    child[i] = parent2[i]
            return child

        def mutate_de(current, candidates, F, CR):
            mutated = np.copy(current)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)
                    F_i = F + 0.1 * np.random.randn()
                    F_i = max(0.1, min(0.9, F_i))  # Dynamic adjustment of F_i based on population diversity
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
                # GA Crossover with dynamic probability
                parent1 = population[i]
                parent2 = population[np.random.choice(pop_size)]
                diversity = np.std(population)  # Measure the diversity of the population
                population[i] = crossover_ga(parent1, parent2, diversity)

                # Novel Adaptive Mutation Strategy
                mutated = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        a, b, c = np.random.choice(pop_size, 3, replace=False)
                        F_j = 0.5 + 0.4 * np.abs(func(candidates[a]) - func(candidates[b]))  # Adaptive mutation based on individual fitness difference
                        mutated[j] = target_to_bounds(candidates[a, j] + F_j * (candidates[b, j] - candidates[c, j]), -5.0, 5.0)
                population[i] = mutated

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