import numpy as np

class ImprovedHarmonySearchWithDE(HarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < pitch_rate:
                    new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
            if func(new_harmony) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            pitch_rate = max(0.1, pitch_rate * 0.99)  # Dynamic pitch adjustment
            
            # Local search step to exploit promising regions
            for i in range(self.dim):
                new_harmony_local = np.copy(self.harmony_memory[-1])
                new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - 0.1),
                                                         min(self.upper_bound, self.harmony_memory[-1][i] + 0.1))
                if func(new_harmony_local) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = new_harmony_local
            
            # Differential Evolution step
            candidate = np.copy(self.harmony_memory[-1])
            mutant = np.random.choice(self.harmony_memory)
            crossover_mask = np.random.choice([0, 1], size=self.dim)
            candidate[crossover_mask == 1] = mutant[crossover_mask == 1]
            if func(candidate) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = candidate
        return self.harmony_memory[0]