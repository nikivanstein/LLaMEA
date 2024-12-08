import numpy as np
from multiprocessing import Pool

class ParallelEvaluationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def evaluate_solution(self, func, solution):
        return func(solution)

    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        mutation_rate = 0.2

        with Pool(processes=4) as pool:
            for _ in range(self.budget - pop_size):
                diversity = np.mean(np.linalg.norm(pop - np.mean(pop, axis=0), axis=1))
                mutation_rate = mutation_rate * (1 + 0.01 * np.exp(-diversity))

                offspring = pool.starmap(self.evaluate_solution, [(func, pop[i] + mutation_rate * np.random.normal(size=self.dim)) for i in range(pop_size)])

                offspring_fitness = np.array(offspring)
                idx = np.argmin(offspring_fitness)
                if offspring_fitness[idx] < fitness[i]:
                    pop[i] = offspring[idx]
                    fitness[i] = offspring_fitness[idx]
                    if offspring_fitness[idx] < func(best_solution):
                        best_solution = offspring[idx]
        
        return best_solution