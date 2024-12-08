import numpy as np

class SelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        scaling_factors = np.full(pop_size, 0.5)
        crossover_probs = np.full(pop_size, 0.5)
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            for idx in range(pop_size):
                target_idx = np.random.choice([i for i in range(pop_size) if i != idx])
                donor_idx1, donor_idx2 = np.random.choice([i for i in range(pop_size) if i not in [idx, target_idx]], size=2, replace=False)
                
                mutated_solution = population[idx] + scaling_factors[idx] * (population[donor_idx1] - population[donor_idx2])
                crossover_prob = np.clip(crossover_probs[idx], 0, 1)
                trial_solution = np.where(np.random.uniform(0, 1, self.dim) < crossover_prob, mutated_solution, population[idx])
                
                trial_fitness = func(trial_solution)
                if trial_fitness < fitness_values[idx]:
                    population[idx] = trial_solution
                    fitness_values[idx] = trial_fitness
                    scaling_factors[idx] *= 1.1
                    crossover_probs[idx] *= 1.1
                else:
                    scaling_factors[idx] *= 0.9
                    crossover_probs[idx] *= 0.9
                
                if trial_fitness < best_fitness:
                    best_solution = trial_solution
                    best_fitness = trial_fitness
        
        return best_solution