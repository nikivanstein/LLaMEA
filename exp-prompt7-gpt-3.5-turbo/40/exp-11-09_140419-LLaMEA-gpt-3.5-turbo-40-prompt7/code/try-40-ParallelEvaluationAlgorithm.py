import numpy as np
import concurrent.futures

class ParallelEvaluationAlgorithm(DynamicMutationAlgorithmSpeed):
    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        mutation_rate = 0.2

        for _ in range(int(self.budget - pop_size)):
            diversity = np.mean(np.linalg.norm(pop - np.mean(pop, axis=0), axis=1))
            mutation_rate = mutation_rate * (1 + 0.01 * np.exp(-diversity))

            offspring = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [executor.submit(self.evaluate_individual, func, pop[i]) for i in range(pop_size)]
                offspring_fitness = np.array([result.result() for result in results])

            for i in range(pop_size):
                mutant = pop[i] + mutation_rate * np.random.normal(size=self.dim)
                offspring.append(mutant)

            idx = np.argmin(offspring_fitness)
            if offspring_fitness[idx] < fitness[i]:
                pop[i] = offspring[idx]
                fitness[i] = offspring_fitness[idx]
                if offspring_fitness[idx] < func(best_solution):
                    best_solution = offspring[idx]
            
            if np.random.rand() < 0.1:  # Randomly increase population size
                new_pop = np.random.uniform(-5.0, 5.0, (5, self.dim))
                new_fitness = np.array([func(ind) for ind in new_pop])
                replace_idx = np.argmax(fitness)
                if new_fitness.min() < fitness[replace_idx]:
                    pop[replace_idx] = new_pop[np.argmin(new_fitness)]
                    fitness[replace_idx] = new_fitness.min()
        
        return best_solution

    def evaluate_individual(self, func, individual):
        return func(individual)