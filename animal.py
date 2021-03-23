import numpy as np


freq_num = 10
output_num = 14
std_gene = 1
std_sense = 1
breeding_age = 5
center = 1
cur_id = 0


def chooser(a):
    r = a > center
    return r


class Animal:
    def __init__(self, x, y):
        global cur_id
        self.id = cur_id
        self.prev_si = None
        cur_id += 1
        self.obj = []
        self.alive = True
        self.hp = 100
        self.x, self.y = x, y
        self.genes = np.zeros((output_num, freq_num * 4))
        self.sound_sense = np.zeros(freq_num)
        self.age = 0
        self.sound = None
        self.true_age = 0

    def make_decision(self, sound_inputs):
        fsi = sound_inputs.flatten()
        outs = np.matmul(self.genes, fsi)
        assert len(outs.shape) == 1 and outs.shape[0] == output_num
        return outs

    @staticmethod
    def breed(a1, a2, c):
        # a1_nbias = 0
        # a1_nbias = float(a1.true_age)/float(a1.true_age + a2.true_age)
        # a1_nbias = 1/(1 + np.exp(-0.02 * (a1.true_age - a2.true_age)))
        a1_nbias = 0.5
        if a1.age < breeding_age or a2.age < breeding_age:
            return False
        for i in range(a1.genes.shape[0]):
            for j in range(a2.genes.shape[1]):
                m = a1.genes[i, j] if np.random.random() > a1_nbias else a2.genes[i, j]
                c.genes[i, j] = np.random.normal(m, std_gene)

        for i in range(a1.sound_sense.shape[0]):
            m = a1.sound_sense[i] if np.random.random() > a1_nbias else a2.sound_sense[i]
            c.sound_sense[i] = max(np.random.normal(m, std_sense), 0.5)

        return True
