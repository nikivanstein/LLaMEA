import numpy as np

class HybridPSOALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 5 * self.dim
        self.inertia_weight = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Update Velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - population[i]) +
                                 self.c2 * r2 * (global_best - population[i]))
                # Update Positions
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate
                fitness = func(population[i])
                evaluations += 1

                # Update Personal and Global Bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness

                # Adaptive Stochastic Local Search
                if np.random.rand() < 0.3:  # 30% chance to refine the solution
                    population[i] = self.adaptive_stochastic_local_search(population[i], func, evaluations / self.budget)
                    fitness = func(population[i])
                    evaluations += 1

        return global_best

    def adaptive_stochastic_local_search(self, solution, func, progress):
        # Adaptive Stochastic Local Search: dynamically adjusts perturbation size
        step_size = 0.05 * (self.upper_bound - self.lower_bound) * (1 - progress ** 2)
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(3):  # Perform a small number of local steps
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)

            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness
        
        return best_solution