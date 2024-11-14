import numpy as np

class EnhancedHybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def _clip(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)
    
    def _initialize_population(self, pop_size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
    
    def _adaptive_parameters(self, eval_count):
        F = 0.5 - 0.2 * (eval_count / self.budget)  # Adjusted mutation factor for better exploration
        Cr = 0.9  # Fixed crossover probability for consistency
        return F, Cr
    
    def _differential_evolution(self, func, population, scores, eval_count):
        pop_size = len(population)
        F, Cr = self._adaptive_parameters(eval_count)
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            rand_indices = np.random.choice(indices, 3, replace=False)
            a, b, c = population[rand_indices]
            mutant = a + F * (b - c)
            mutant = self._clip(mutant)
            
            crossover_mask = np.random.rand(self.dim) < Cr
            trial = np.where(crossover_mask, mutant, population[i])
            trial_score = func(trial)
            
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score
        return population, scores
    
    def _simulated_annealing(self, func, x, score):
        temperature = 1.0
        cooling_rate = 0.95  # Adjusted cooling rate for gradual temperature decrease
        for _ in range(25):  # Slightly more iterations
            neighbor = x + np.random.normal(scale=0.1, size=self.dim)
            neighbor = self._clip(neighbor)
            neighbor_score = func(neighbor)
            
            if neighbor_score < score or np.random.rand() < np.exp((score - neighbor_score) / temperature):
                x, score = neighbor, neighbor_score
            
            temperature *= cooling_rate
        return x, score
    
    def __call__(self, func):
        pop_size = min(20, self.budget // 4)  # Adjusted population size
        population = self._initialize_population(pop_size)
        scores = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        while eval_count < self.budget:
            population, scores = self._differential_evolution(func, population, scores, eval_count)
            best_idx = np.argmin(scores)
            best_individual, best_score = population[best_idx], scores[best_idx]
            
            best_individual, best_score = self._simulated_annealing(func, best_individual, best_score)
            
            population[best_idx] = best_individual
            scores[best_idx] = best_score
            eval_count += (25 // 2)  # Updated for revised SA evaluation count
            
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]