import numpy as np

class HybridGADifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(20, dim * 5)  # Dynamic population size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.tournament_size = 3
        self.improvement_history = []  # Track improvements

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def tournament_selection(self, population, fitness):
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        best_index = indices[np.argmin(fitness[indices])]
        return population[best_index]

    def differential_mutation(self, target_idx, population):
        idxs = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        scale = 0.6 + 0.4 * (self.evaluations / self.budget)  # Changed scaling factor
        mutant = population[a] + scale * (population[b] - population[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_rate = 0.7 + 0.3 * (1 - np.exp(-5 * (self.budget - self.evaluations) / self.budget))  # Changed cross_rate
        cross_points = np.random.rand(self.dim) < cross_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, solution, func):
        if self.improvement_history and np.mean(self.improvement_history[-5:]) < 1e-5:
            return solution  # Skip if no recent improvement
        learning_rate = np.exp(-5 * self.evaluations / self.budget)  # New adaptive learning rate
        perturbation = np.random.normal(0, 0.05 * learning_rate, self.dim)  # Modified perturbation with learning rate
        neighbor = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
        return neighbor if func(neighbor) < func(solution) else solution

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.evaluations = self.population_size
        best_global = population[np.argmin(fitness)].copy()
        best_global_fitness = func(best_global)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                parent = self.tournament_selection(population, fitness)
                self.mutation_factor = 0.5 + 0.3 * (1 - np.exp(-5 * self.evaluations / self.budget))
                mutant = self.differential_mutation(i, population)
                trial = self.crossover(parent, mutant)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    improvement = fitness[i] - trial_fitness
                    self.improvement_history.append(improvement)  # Track improvements
                    if trial_fitness < best_global_fitness:
                        best_global = trial.copy()
                        best_global_fitness = trial_fitness
                        best_global = self.local_search(best_global, func)  # Apply local search

                if self.evaluations >= self.budget:
                    break

        return best_global