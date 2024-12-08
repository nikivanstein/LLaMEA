import numpy as np

class Heap:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.score = -np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            if len(self.population) < self.dim:
                # Initialize individual with random search space
                individual = np.random.uniform(-5.0, 5.0, self.dim)
            else:
                # Select parent with highest score
                parent_idx = np.argmax([func(individual) for individual in self.population])
                parent = self.population[parent_idx]

                # Refine parent's strategy with adaptive probability
                parent_idx = np.random.choice(len(self.population), p=[0.7, 0.3])
                parent = self.population[parent_idx]

                # Create offspring by perturbing parent
                offspring = parent + np.random.uniform(-0.5, 0.5, self.dim)

                # Ensure search space bounds
                offspring = np.clip(offspring, -5.0, 5.0)

            # Evaluate offspring
            score = func(offspring)

            # Update population and score
            self.population.append(offspring)
            self.score = max(self.score, score)

            # Replace worst individual if necessary
            if len(self.population) > self.dim:
                self.population.remove(min(self.population, key=lambda x: func(x)))

    def get_best(self):
        return self.population[np.argmax([func(individual) for individual in self.population])]

# Example usage
def func(x):
    return np.sum(x**2)

heap = Heap(100, 10)
heap(0, func)
print(heap.get_best())