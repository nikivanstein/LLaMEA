class DynamicMemorySizeHarmonySearch(HarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        memory_size = int(0.1 * self.budget)  # Dynamic memory size adjustment
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for i in range(self.dim):
                if np.random.rand() < pitch_rate:
                    new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
            if func(new_harmony) < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmony
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            pitch_rate = max(0.1, pitch_rate * 0.99 * 1.087)
            for i in range(self.dim):
                new_harmony_local = np.copy(self.harmony_memory[-1])
                new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - 0.1),
                                                         min(self.upper_bound, self.harmony_memory[-1][i] + 0.1))
                if func(new_harmony_local) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = new_harmony_local
            if _ % (self.budget // memory_size) == 0 and memory_size < len(self.harmony_memory):
                self.harmony_memory = self.harmony_memory[:memory_size]
        return self.harmony_memory[0]