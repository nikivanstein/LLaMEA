import numpy as np
import matplotlib.pyplot as plt

class NoisyBlackBoxOptimizer:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.explore_eviction = False
        self.current_dim = 0
        self.func = None

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Gradient descent with hierarchical clustering for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
                    if self.current_dim == 0:
                        # Gradient descent without hierarchical clustering
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                    else:
                        # Hierarchical clustering to select the best function to optimize
                        cluster_labels = np.argpartition(func, self.current_dim)[-1]
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
                        self.current_dim += 1
                        if self.budget == 0:
                            break
                self.budget -= 1
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

def evaluate_bbof(func, x, y, z):
    return np.mean((func(x) - func(y) - func(z) + 0.5) ** 2)

def mutate(individual, dim):
    return np.random.uniform(-5.0, 5.0, dim)

def selection(population, tournament_size):
    winners = []
    for _ in range(tournament_size):
        tournament = np.random.choice(population, size=population.size, replace=False)
        winners.append(np.mean(evaluate_bbof(func, *tournament), axis=0))
    return np.array(winners)

def evolutionary_algorithm(func, budget, dim, max_iter):
    population = [np.array([func(np.random.uniform(-5.0, 5.0, dim))]) for _ in range(100)]
    for _ in range(max_iter):
        new_population = []
        for _ in range(population.size // 2):
            parent1, parent2 = population[np.random.randint(0, population.size, size=2)]
            child = (parent1 + parent2) / 2
            if np.random.rand() < 0.5:
                child = mutate(child, dim)
            new_population.append(child)
        population = new_population
        # Hierarchical clustering to select the best function to optimize
        cluster_labels = np.argpartition(func, dim)[-1]
        new_population = [func(x) for x in np.random.uniform(-5.0, 5.0, dim) if cluster_labels == cluster_labels]
        population = new_population
    return population

def main():
    budget = 1000
    dim = 5
    max_iter = 1000
    population = evolutionary_algorithm(func, budget, dim, max_iter)
    # Update the individual lines of the selected solution to refine its strategy
    for i in range(population.size):
        if population[i] < -5.0:
            population[i] = -np.inf
        elif population[i] > 5.0:
            population[i] = np.inf
    # Plot the results
    plt.plot([np.mean(evaluate_bbof(func, *individual)) for individual in population])
    plt.show()

if __name__ == "__main__":
    main()