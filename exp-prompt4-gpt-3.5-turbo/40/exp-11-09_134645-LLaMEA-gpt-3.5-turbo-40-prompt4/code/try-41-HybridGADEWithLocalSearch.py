import numpy as np

class HybridGADEWithLocalSearch:
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

        def crossover_ga(parent1, parent2):
            child = np.copy(parent1)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    child[i] = parent2[i]
            return child

        def mutate_levy(current):
            mutated = np.copy(current)
            beta = 1.5
            scale = 0.01
            levy = np.random.standard_t(beta, size=self.dim) * scale
            mutated = target_to_bounds(mutated + levy, -5.0, 5.0)
            return mutated

        def global_best_pso(particles, best_particle):
            return best_particle

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget // pop_size):
            population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            velocities = np.zeros((pop_size, self.dim))
            best_particle = np.copy(best_solution)
            for i in range(pop_size):
                parent1 = population[i]
                parent2 = population[np.random.choice(pop_size)]
                population[i] = crossover_ga(parent1, parent2)

                trial_mutate = mutate_levy(population[i])
                mutated_fitness = func(trial_mutate)
                if mutated_fitness < func(population[i]):
                    population[i] = trial_mutate

                velocities[i] = omega * velocities[i] + c1 * np.random.rand(self.dim) * (best_particle - population[i]) + c2 * np.random.rand(self.dim) * (best_solution - population[i])
                population[i] = target_to_bounds(population[i] + velocities[i], -5.0, 5.0)

                population[i] = local_search(population[i], func)

                if func(population[i]) < best_fitness:
                    best_solution = population[i]
                    best_fitness = func(population[i])
                best_particle = global_best_pso(population, best_particle)

        return best_solution