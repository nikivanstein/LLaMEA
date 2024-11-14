import numpy as np
import concurrent.futures

class ImprovedDynamicMutationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def evaluate_solution(self, func, pop):
        return func(pop)

    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        mutation_rate = 0.2

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(self.budget - pop_size):
                diversity = np.mean(np.linalg.norm(pop - np.mean(pop, axis=0), axis=1))
                mutation_rate = mutation_rate * (1 + 0.01 * np.exp(-diversity))

                offspring = []
                futures = {executor.submit(self.evaluate_solution, func, pop[i] + mutation_rate * np.random.normal(size=self.dim)): i for i in range(pop_size)}
                concurrent.futures.wait(futures)

                for future in futures:
                    i = futures[future]
                    offspring_fitness = future.result()
                    if offspring_fitness < fitness[i]:
                        pop[i] = pop[i] + mutation_rate * np.random.normal(size=self.dim)
                        fitness[i] = offspring_fitness
                        if offspring_fitness < func(best_solution):
                            best_solution = pop[i]

        return best_solution