import numpy as np

class ChaoticHybridDEALS:
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
        chaotic_seq = self.chaotic_map(self.population_size)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution Mutation with chaotic influence
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                chaotic_factor = chaotic_seq[evaluations % len(chaotic_seq)]
                mutant = np.clip(x0 + self.F * chaotic_factor * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Adaptive Stochastic Local Search
                if np.random.rand() < 0.25:  # Chance to refine the trial solution
                    trial = self.adaptive_stochastic_local_search(trial, func, evaluations / self.budget)

                # Evaluate Trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)]

    def adaptive_stochastic_local_search(self, solution, func, progress):
        # Adaptive Stochastic Local Search: dynamically adjusts perturbation size
        step_size = 0.1 * (self.upper_bound - self.lower_bound) * (1 - progress ** 2)
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

    def chaotic_map(self, size):
        # Generate a chaotic sequence using Logistic map
        x = 0.7
        chaotic_sequence = []
        for _ in range(size):
            x = 3.7 * x * (1 - x)
            chaotic_sequence.append(x)
        return chaotic_sequence