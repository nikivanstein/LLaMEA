class Enhanced_Improved_EA_ADES(Improved_EA_ADES):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                # Existing code for mutation and crossover
                # Dynamically adjust mutation and crossover probabilities based on individual performance
                if trial_fit < fitness[i]:
                    self.mut_prob = min(1.0, self.mut_prob * 1.1)
                    self.cross_prob = max(0.1, self.cross_prob * 0.9)
                else:
                    self.mut_prob = max(0.1, self.mut_prob * 0.9)
                    self.cross_prob = min(1.0, self.cross_prob * 1.1)
                new_pop[i] = pop[i] if trial_fit >= fitness[i] else trial
            pop = new_pop