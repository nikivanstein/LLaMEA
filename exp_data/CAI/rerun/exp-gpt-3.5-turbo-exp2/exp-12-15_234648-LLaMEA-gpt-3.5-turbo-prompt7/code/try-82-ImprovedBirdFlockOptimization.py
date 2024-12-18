class ImprovedBirdFlockOptimization:
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))

        def fitness(position):
            return func(position)

        def update_velocity(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget) * (1.0 - 0.5 * iteration / self.budget)  # Dynamic inertia weight update
            chaos_map = lambda x: 4 * x * (1 - x)  # Logistic chaotic map
            chaotic_values = chaos_map(np.random.rand(self.dim))
            return w * velocity + self.c1 * r1 * chaotic_values * (personal_best_pos - position) + self.c2 * r2 * chaotic_values * (global_best_pos - position)
        
        def chaotic_de_step(population, velocity, personal_best_pos, global_best_pos, itr):
            F, CR = 0.5 + 0.3 * np.random.rand(), 0.1 + 0.9 * np.random.rand()
            new_population = np.copy(population)
            for i in range(self.num_birds):
                idxs = [idx for idx in range(self.num_birds) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                trial = np.where(np.random.uniform(0, 1, self.dim) < CR, mutant, population[i])
                if fitness(trial) < fitness(new_population[i]):
                    new_population[i] = trial
            return new_population

        population = initialize_population()
        velocity = np.zeros((self.num_birds, self.dim))
        personal_best_pos = population.copy()
        global_best_pos = personal_best_pos[np.argmin([fitness(ind) for ind in personal_best_pos])

        for itr in range(self.budget):
            population = chaotic_de_step(population, velocity, personal_best_pos, global_best_pos, itr)
            personal_best_pos = np.array([ind if fitness(ind) < fitness(personal_best_pos[i]) else personal_best_pos[i] for i, ind in enumerate(population)])
            global_best_pos = personal_best_pos[np.argmin([fitness(ind) for ind in personal_best_pos])]

        return global_best_pos