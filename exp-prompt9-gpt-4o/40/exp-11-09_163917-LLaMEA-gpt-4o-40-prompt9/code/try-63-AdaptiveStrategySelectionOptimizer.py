import numpy as np

class AdaptiveStrategySelectionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(150, budget // 3)
        self.mutation_factor_base = 0.8
        self.crossover_prob = 0.85
        self.local_search_prob = 0.25
        self.tournament_size = 4
        self.strategy_probs = [0.5, 0.5]  # Strategy selection probabilities
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                if np.random.rand() < self.strategy_probs[0]:
                    chosen = np.random.choice(indices, self.tournament_size, replace=False)
                    best_idx = min(chosen, key=lambda idx: self.fitness[idx])
                    a, b, c = self.population[best_idx], self.population[np.random.choice(indices)], self.population[np.random.choice(indices)]
                    mutation_factor = self.mutation_factor_base * (1 - self.evaluations / self.budget)
                else:
                    chosen = np.random.choice(indices, self.tournament_size, replace=False)
                    a, b, c = self.population[chosen[0]], self.population[chosen[1]], self.population[chosen[2]]
                    mutation_factor = self.mutation_factor_base * np.random.rand()

                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Enhanced dynamic local search
                if np.random.rand() < self.local_search_prob * (1 - self.evaluations / self.budget):
                    self.enhanced_local_search(i, func)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def enhanced_local_search(self, index, func):
        step_size = 0.05 * (self.upper_bound - self.lower_bound)
        for _ in range(3):
            if self.evaluations >= self.budget:
                break
            
            perturbation = np.random.normal(0, step_size, self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness