import numpy as np

class AdvancedAdaptiveEvolution:
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
        F = 0.5 - 0.3 * (eval_count / self.budget)  # Optimized mutation factor
        Cr = 0.8 + 0.2 * np.sin(eval_count / self.budget * np.pi / 2)  # Sine crossover probability
        return F, Cr
    
    def _differential_evolution(self, func, population, scores, eval_count):
        pop_size = len(population)
        F, Cr = self._adaptive_parameters(eval_count)
        for i in range(pop_size):
            indices = np.delete(np.arange(pop_size), i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = self._clip(a + F * (b - c))  # Directly clip mutant
            crossover_mask = np.random.rand(self.dim) < Cr
            trial = np.where(crossover_mask, mutant, population[i])
            trial_score = func(trial)
            
            if trial_score < scores[i]:
                population[i], scores[i] = trial, trial_score
        return population, scores
    
    def _simulated_annealing(self, func, x, score):
        temperature = 1.0
        cooling_rate = 0.9  # Optimized cooling rate
        iteration_limit = max(10, self.budget // 100)  # More iterations for thorough search
        for _ in range(iteration_limit):
            neighbor = self._clip(x + np.random.normal(0, 0.1, self.dim))  # Adaptive perturbation
            neighbor_score = func(neighbor)
            
            if neighbor_score < score or np.random.rand() < np.exp((score - neighbor_score) / temperature):
                x, score = neighbor, neighbor_score
            
            temperature *= cooling_rate
        return x, score
    
    def __call__(self, func):
        pop_size = min(50, self.budget // 4)  # Increased population size for diversity
        population = self._initialize_population(pop_size)
        scores = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        while eval_count < self.budget:
            population, scores = self._differential_evolution(func, population, scores, eval_count)
            best_idx = np.argmin(scores)
            best_individual, best_score = population[best_idx], scores[best_idx]
            
            best_individual, best_score = self._simulated_annealing(func, best_individual, best_score)
            
            population[best_idx], scores[best_idx] = best_individual, best_score
            eval_count += min(pop_size, self.budget - eval_count)  # Ensure budget compliance
            
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]