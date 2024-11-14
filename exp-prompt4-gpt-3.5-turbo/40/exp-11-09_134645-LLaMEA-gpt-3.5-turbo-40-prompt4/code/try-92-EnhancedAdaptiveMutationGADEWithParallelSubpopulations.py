import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedAdaptiveMutationGADEWithParallelSubpopulations:
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

        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in range(self.budget // (initial_pop_size * 4)):
                populations = [np.random.uniform(-5.0, 5.0, (initial_pop_size, self.dim)) for _ in range(4)]
                velocities = [np.zeros((initial_pop_size, self.dim)) for _ in range(4)]
                best_particles = [np.copy(best_solution) for _ in range(4)]
                fitness_arrays = [np.zeros(initial_pop_size) for _ in range(4)]

                for k in range(4):
                    for i in range(initial_pop_size):
                        parent1 = populations[k][i]
                        parent2 = populations[k][np.random.choice(initial_pop_size)]
                        diversity = np.std(populations[k])
                        populations[k][i] = crossover_ga(parent1, parent2, diversity)

                        trial_de = mutate_de(populations[k][i], populations[k], F_de, CR_de, fitness_arrays[k])
                        mutated_fitness_de = func(trial_de)
                        fitness_arrays[k][i] = mutated_fitness_de
                        if mutated_fitness_de < func(populations[k][i]):
                            populations[k][i] = trial_de

                        velocities[k][i] = omega * velocities[k][i] + c1 * np.random.rand(self.dim) * (best_particles[k] - populations[k][i]) + c2 * np.random.rand(self.dim) * (best_solution - populations[k][i])
                        populations[k][i] = target_to_bounds(populations[k][i] + velocities[k][i], -5.0, 5.0)

                        populations[k] = differential_evolution(populations[k], func)

                        populations[k][i] = local_search(populations[k][i], func)

                        if func(populations[k][i]) < best_fitness:
                            best_solution = populations[k][i]
                            best_fitness = func(populations[k][i])
                        best_particles[k] = global_best_pso(populations[k], best_particles[k])

        return best_solution