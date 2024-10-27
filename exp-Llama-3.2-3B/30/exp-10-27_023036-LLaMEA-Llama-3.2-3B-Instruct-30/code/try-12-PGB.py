import numpy as np
from scipy.optimize import minimize

class PGB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.eval_func = None

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        # Evaluate the objective function at each point in the population
        scores = [func(point) for point in population]
        for i, (point, score) in enumerate(zip(population, scores)):
            self.population.append((f"Point {i}", "Random initialization", score))
        
        # Select the best point in the population
        best_point = population[np.argmin(scores)]
        self.population.append((f"Best point so far", "Random initialization", scores[np.argmin(scores)]))
        
        # Refine the search strategy using gradient boosting
        for _ in range(self.budget - 1):
            # Select a subset of the population to refine
            subset = np.random.choice(len(self.population), int(self.budget * 0.3), replace=False)
            subset_points = [self.population[i][0] for i in subset]
            subset_scores = [self.population[i][2] for i in subset]
            
            # Refine the search strategy using gradient boosting
            refined_points = []
            for _ in range(int(self.budget * 0.7)):
                # Select a point to refine
                idx = np.random.choice(subset_points)
                point = self.population[idx][0]
                
                # Refine the point using gradient boosting
                gradient = np.gradient(func(point))
                refined_point = point + 0.1 * gradient
                refined_points.append(refined_point)
            
            # Evaluate the objective function at the refined points
            refined_scores = [func(point) for point in refined_points]
            for i, (point, score) in enumerate(zip(refined_points, refined_scores)):
                self.population.append((f"Refined point {i}", "Gradient boosting", score))
            
            # Update the best point in the population
            scores = [func(point) for point in refined_points]
            best_point = refined_points[np.argmin(scores)]
            self.population.append((f"Best point after refinement", "Gradient boosting", scores[np.argmin(scores)]))
        
        # Refine the search strategy using probability-based mutation
        for _ in range(self.budget - 1):
            # Select a point to refine
            idx = np.random.choice(len(self.population))
            point = self.population[idx][0]
            
            # Refine the point using probability-based mutation
            mutation_prob = np.random.rand()
            if mutation_prob < 0.3:
                # Apply a random mutation to the point
                mutation = np.random.uniform(-0.1, 0.1, self.dim)
                point += mutation
            self.population.append((f"Refined point after mutation", "Probability-based mutation", func(point)))
        
        # Return the best point in the population
        best_point = population[np.argmin([func(point) for point in population])]
        return best_point

# Example usage:
def func(x):
    return np.sum(x**2)

pgb = PGB(100, 10)
best_point = pgb(func)
print(best_point)