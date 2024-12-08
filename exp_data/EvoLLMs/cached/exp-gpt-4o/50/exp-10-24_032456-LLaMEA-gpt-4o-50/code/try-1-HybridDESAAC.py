import numpy as np

class HybridDESAAC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(0.5 * dim)
        self.prob_crossover = 0.9
        self.F = 0.8  # Differential weight
        self.current_evaluations = 0
        self.adaptive_rate = 0.2  # Adaptive adjustment factor

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.current_evaluations < self.budget:
            # Adaptively adjust crossover probability
            self.prob_crossover = max(0.1, self.prob_crossover - self.adaptive_rate * (self.current_evaluations / self.budget))
            
            # Differential Evolution Mutation
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break
                
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.prob_crossover
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                self.current_evaluations += 1
                delta_e = trial_fitness - fitness[i]
                acceptance_probability = np.exp(-delta_e / (1 + self.current_evaluations / self.budget))
                
                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

        return best_solution

# Example of instantiation and usage:

# optimizer = HybridDESAAC(budget=2000, dim=10)
# best_solution = optimizer(my_black_box_function)