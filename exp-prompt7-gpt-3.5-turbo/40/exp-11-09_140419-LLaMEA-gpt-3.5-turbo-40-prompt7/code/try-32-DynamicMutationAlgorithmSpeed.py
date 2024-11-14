import numpy as np

class DynamicMutationAlgorithmSpeed:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        mutation_rate = 0.2

        for _ in range(int(self.budget - pop_size)):
            diversity = np.mean(np.linalg.norm(pop - np.mean(pop, axis=0), axis=1))
            mutation_rate = mutation_rate * np.exp(-0.02 * diversity)

            offspring = []
            for i in range(pop_size):
                mutant = pop[i] + mutation_rate * np.random.normal(size=self.dim)
                offspring.append(mutant)

            offspring_fitness = np.array([func(ind) for ind in offspring])
            idx = np.argmin(offspring_fitness)
            mask = offspring_fitness < fitness
            pop[mask] = offspring[mask]
            fitness[mask] = offspring_fitness[mask]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best_solution):
                best_solution = pop[best_idx]

            if np.random.rand() < 0.1:  # Randomly increase population size
                new_pop = np.random.uniform(-5.0, 5.0, (5, self.dim))
                new_fitness = np.array([func(ind) for ind in new_pop])
                replace_idx = np.argmax(fitness)
                replace_mask = new_fitness < fitness[replace_idx]
                pop[replace_mask] = new_pop[replace_mask]
                fitness[replace_mask] = new_fitness[replace_mask]

        return best_solution