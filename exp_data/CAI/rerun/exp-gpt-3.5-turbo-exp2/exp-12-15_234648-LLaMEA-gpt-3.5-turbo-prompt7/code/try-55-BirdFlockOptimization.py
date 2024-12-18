class BirdFlockOptimization:
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5, a=2.0):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.a = a

    def __call__(self, func):
        def update_velocity(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget)  # Dynamic inertia weight
            a = self.a * (1.0 - iteration / self.budget)  # Dynamic acceleration coefficient
            return w * velocity + a * (self.c1 * r1 * (personal_best_pos - position) + self.c2 * r2 * (global_best_pos - position))