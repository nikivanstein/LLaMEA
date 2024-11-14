import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.local_perturbation_rate = 0.2  # local perturbation for intensification

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluation
                trial_fitness = func(trial)
                num_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                
                if num_evaluations >= self.budget:
                    break

            # Local Perturbation on best solutions to enhance exploration
            num_perturbations = int(self.local_perturbation_rate * self.population_size)
            top_indices = np.argsort(fitness)[:num_perturbations]
            for i in top_indices:
                if num_evaluations >= self.budget:
                    break
                candidate = population[i] + np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = func(candidate)
                num_evaluations += 1
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness
        
        return best_solution, best_fitness