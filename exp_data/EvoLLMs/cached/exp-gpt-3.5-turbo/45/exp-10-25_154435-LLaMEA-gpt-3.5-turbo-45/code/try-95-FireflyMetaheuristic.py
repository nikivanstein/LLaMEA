class FireflyMetaheuristic:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, x, y):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x - y))

    def move_firefly(self, x, y, attractiveness):
        return x + self.alpha * (y - x) + attractiveness * np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        current_solution = np.random.uniform(-5.0, 5.0, self.dim)
        for _ in range(self.budget):
            for _ in range(self.budget):
                new_solution = self.move_firefly(current_solution, np.random.uniform(-5.0, 5.0, self.dim), self.attractiveness(current_solution, new_solution))
                if func(new_solution) < func(current_solution):
                    current_solution = new_solution
        return current_solution