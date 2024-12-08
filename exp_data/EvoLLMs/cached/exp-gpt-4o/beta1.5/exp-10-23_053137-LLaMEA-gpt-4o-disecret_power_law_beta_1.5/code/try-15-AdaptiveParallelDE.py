import numpy as np
import concurrent.futures

class AdaptiveParallelDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 5 * self.dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(self.population_size):
                    if evaluations >= self.budget:
                        break

                    futures.append(executor.submit(self.evolve, population, fitness, i, func, evaluations))

                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res:
                        population[res[0]], fitness[res[0]], evaluations = res[1], res[2], res[3]

        return population[np.argmin(fitness)]

    def evolve(self, population, fitness, i, func, evaluations):
        indices = np.random.choice(self.population_size, 3, replace=False)
        x0, x1, x2 = population[indices]
        mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)

        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, population[i])

        if np.random.rand() < 0.3:  # 30% chance to refine the trial solution
            trial = self.dynamic_local_search(trial, func, evaluations / self.budget)

        trial_fitness = func(trial)
        evaluations += 1

        if trial_fitness < fitness[i]:
            return i, trial, trial_fitness, evaluations
        else:
            return None

    def dynamic_local_search(self, solution, func, progress):
        step_size = 0.1 * (self.upper_bound - self.lower_bound) * np.exp(-progress)
        best_solution = solution.copy()
        best_fitness = func(best_solution)

        for _ in range(5):  # Perform a few more local steps
            perturbation = np.random.normal(0, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)

            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness

        return best_solution