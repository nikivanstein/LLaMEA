import numpy as np

class EnhancedHybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)  # Increased population size for diversity
        self.mutation_factor = 0.9  # Adjusted mutation factor for better exploration
        self.crossover_prob = 0.85  # Adjusted crossover probability
        self.local_search_prob = 0.4  # Increased local search probability for frequent refinements
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Enhanced adaptive local search with dynamic step size
                if np.random.rand() < self.local_search_prob:
                    self.enhanced_local_search(i, func)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def enhanced_local_search(self, index, func):
        step_size = 0.05 * (self.upper_bound - self.lower_bound)  # Reduced step size for precision
        for _ in range(7):  # Increased number of local perturbations
            if self.evaluations >= self.budget:
                break
            
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness