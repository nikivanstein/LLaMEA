import numpy as np

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class MGDALRMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        mgdalr = MGDALR(self.budget, self.dim)
        return mgdalr(func)

class MGDALRWithProbabilities(MGDALRMetaheuristic):
    def __init__(self, budget, dim, probabilities):
        super().__init__(budget, dim)
        self.probabilities = probabilities

    def __call__(self, func):
        mgdalr = MGDALR(self.budget, self.dim)
        probabilities = self.probabilities / sum(self.probabilities)
        return mgdalr(func, probabilities)

class MGDALRWithBayes(MGDALRMetaheuristic):
    def __init__(self, budget, dim, prior):
        super().__init__(budget, dim)
        self.prior = prior

    def __call__(self, func):
        mgdalr = MGDALR(self.budget, self.dim)
        probabilities = self.prior / sum(self.prior)
        return mgdalr(func, probabilities)

# Example usage:
def f(x):
    return np.sum(x**2)

probabilities = [0.1, 0.2, 0.7]  # probabilities of each direction
prior = 1.0 / len(probabilities)  # prior distribution

mgdalr_with_probabilities = MGDALRWithProbabilities(budget=100, dim=10, probabilities=probabilities)
mgdalr_with_bayes = MGDALRWithBayes(budget=100, dim=10, prior=np.exp(np.linspace(0, 10, 10)))

mgdalr_with_probabilities(func=f, x=[-5.0] * 10)
mgdalr_with_bayes(func=f, x=[-5.0] * 10)