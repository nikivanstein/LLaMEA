import numpy as np

class ProbabilityPerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.best_solution = None
        self.best_score = -np.inf

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)

        # Evaluate population
        scores = [func(solution) for solution in self.population]

        # Update best solution
        self.best_solution = np.argmax(scores)
        self.best_score = scores[self.best_solution]

        # Perturb solutions based on probability
        perturbed_population = []
        for i in range(self.budget):
            if np.random.rand() < 0.3:
                # Perturb x-axis
                perturbation = np.random.uniform(-1.0, 1.0)
                perturbed_population.append(self.population[i] + perturbation)
            else:
                perturbed_population.append(self.population[i])

        # Replace worst solution with perturbed solution
        scores = [func(solution) for solution in perturbed_population]
        worst_index = np.argmin(scores)
        self.population[worst_index] = perturbed_population[worst_index]

        # Evaluate perturbed population
        scores = [func(solution) for solution in self.population]

        # Update best solution
        self.best_solution = np.argmax(scores)
        self.best_score = scores[self.best_solution]

# Test the algorithm
budget = 100
dim = 10
func = lambda x: np.sum([i**2 for i in x])  # Example black box function
optimizer = ProbabilityPerturbation(budget, dim)
optimizer(func)
print(optimizer.best_solution)
print(optimizer.best_score)