import numpy as np

class AdaptiveChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        mutation_probs = np.full(self.budget, 0.3)  # Initialize mutation probabilities
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for idx, ind in enumerate(population):
                mutation_prob = mutation_probs[idx]
                if mutation_prob < 0.3:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 0.8, self.dim)  # Adjust mutation range
                elif mutation_prob < 0.6:
                    new_ind = ind + np.random.randn(self.dim) * 0.5  # Scale mutation step
                else:
                    new_ind = np.random.uniform(-5.0, 5.0, self.dim)
                new_population.append(new_ind)
                # Update mutation probability based on individual performance
                if func(new_ind) < func(ind):
                    mutation_probs[idx] = min(0.5, mutation_probs[idx] * 1.2)  # Increase for good performers
                else:
                    mutation_probs[idx] = max(0.1, mutation_probs[idx] * 0.8)  # Decrease for poor performers
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]