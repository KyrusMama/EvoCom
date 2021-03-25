import numpy as np
from sklearn.neural_network import MLPClassifier

freq_num = 10
output_num = 14
std_gene = 1
std_sense = 1
breeding_age = 5
center = 1
cur_id = 0
hidden_state_size = 20

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
        self.network = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20), random_state=1, activation='logistic')
        self.network.fit([np.random.random(freq_num * 4 + hidden_state_size)], [np.random.randint(2,size=output_num + hidden_state_size)])
        self.sound_sense = np.zeros(freq_num)
        self.hidden_state = np.random.random(hidden_state_size)
        self.age = 0
        self.sound = None
        self.true_age = 0

    def make_decision(self, sound_inputs):
        fsi = [np.concatenate([self.hidden_state, sound_inputs.flatten()], axis=0)]
        #outs = np.matmul(self.genes, fsi)
        outs = self.network.predict(fsi)[0]
        self.hidden_state = outs[:hidden_state_size]
        outs = outs[hidden_state_size:]
        assert len(outs.shape) == 1 and outs.shape[0] == output_num
        return outs

    @staticmethod
    def breed(a1, a2, c):
        # a1_nbias = 0
        # a1_nbias = float(a1.true_age)/float(a1.true_age + a2.true_age)
        # a1_nbias = 1/(1 + np.exp(-0.02 * (a1.true_age - a2.true_age)))
        # a1_nbias = 0.5
        if a1.age < breeding_age or a2.age < breeding_age:
            return False


        for i in range(len(a1.network.coefs_)):
            selector = np.random.randint(2, size=a1.network.coefs_[i].shape)
            child_weights = a1.network.coefs_[i] * selector + a2.network.coefs_[i] * (1 - selector) + np.random.normal(0, std_gene, size=a1.network.coefs_[i].shape)
            c.network.coefs_[i] = child_weights

        c.sound_sense = np.zeros(freq_num)
        # for i in range(a1.sound_sense.shape[0]):
        #    m = a1.sound_sense[i] if np.random.random() > a1_nbias else a2.sound_sense[i]
        #    c.sound_sense[i] = max(np.random.normal(m, std_sense), 0.5)

        return True
