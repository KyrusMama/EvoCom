import board
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time
import animal


take_inputs = False


def plotter():
    for i in range(2):
        oneDir = [x[0] for (x,id) in board.move_sound_angle if id == i]
        temp = [np.mean(oneDir[max(i - 100, 0): min(i + 100, len(oneDir) - 1)])
                for i in range(len(oneDir))]
        plt.plot(temp,  color='blue')
        oneDir = [x[1] for (x,id) in board.move_sound_angle if id == i]
        temp = [np.mean(oneDir[max(i - 100, 0): min(i + 100, len(oneDir) - 1)])
                for i in range(len(oneDir))]
        plt.plot(temp, color='red')
        oneDir = [x[2] for (x,id) in board.move_sound_angle if id == i]
        temp = [np.mean(oneDir[max(i - 100, 0): min(i + 100, len(oneDir) - 1)])
                for i in range(len(oneDir))]
        plt.plot(temp, color='green')
        oneDir = [x[3] for (x,id) in board.move_sound_angle if id == i]
        temp = [np.mean(oneDir[max(i - 100, 0): min(i + 100, len(oneDir) - 1)])
                for i in range(len(oneDir))]
        plt.plot(temp, color='orange')
        
        plt.title('angle ' + str(i))
        plt.figure()

    for spec_id in range(board.n_species):
        
        for spec_id2 in range(board.n_species):
            out_lst = []
            for key in saved_networks.keys():
                tspecies_id, tid = key
                if tspecies_id == spec_id:
                    fsi = [np.concatenate([np.zeros(animal.hidden_state_size),
                                           np.zeros(animal.freq_num * 4)], axis=0)]
                    p1 = saved_networks[key].predict(fsi)[0]
                    mem = p1[:animal.hidden_state_size]
                    fsi = [np.concatenate([mem,
                                           np.zeros(animal.freq_num * 4)], axis=0)]
                    p1 = saved_networks[key].predict(fsi)[0]

                    sounds = np.zeros((4, animal.freq_num))
                    sounds[0, spec_id2] = 10.
                    fsi = [np.concatenate([np.zeros(animal.hidden_state_size),
                                           sounds.flatten()], axis=0)]
                    p2 = saved_networks[key].predict(fsi)[0]
                    mem = p2[:animal.hidden_state_size]
                    fsi = [np.concatenate([mem,
                                           sounds.flatten()], axis=0)]
                    p2 = saved_networks[key].predict(fsi)[0]

                    # print(p1, p2)
                    p1_mo = np.exp(p1[animal.hidden_state_size: animal.hidden_state_size + 4])
                    p2_mo = np.exp(p2[animal.hidden_state_size: animal.hidden_state_size + 4])

                    diff = 2 * ((p2_mo / np.sum(p2_mo)) - (p1_mo / np.sum(p1_mo)))
                    dist = diff[0] - diff[2]
                    out_lst.append(dist)

            plt.plot([np.mean(out_lst[max(i - 10, 0): min(i + 10, len(out_lst) - 1)])
                    for i in range(len(out_lst))])
        plt.title('effect of sound on species '+str(spec_id))
        plt.figure()
    plt.show()


if __name__ == "__main__":
    num = 50
    b = board.Board(40, 80)
    b.lay_start(num)
    turns = [a for a in b.animals]
    count = 0
    max_count = 1
    avg_sens = []
    avg_sound = []
    std_sound = []
    avg_hp = []
    avg_sound_genes = []
    std_sound_genes = []

    saved_networks = {}

    while len(turns) > 0:
        if count % (num) == 0:
            print(count)
            if board.shouldDraw:
                b.w.update_idletasks()
                b.w.update()
        count += 1
        if count >= max_count:
            inp = input()
            if "q" in inp:
                break
            elif "p" in inp:
                plotter()
            else:
                try:
                    max_count += int(inp)
                except:
                    max_count += 1
        else:
            time.sleep(0)

        cur = turns.pop(0)
        #print(count, cur.id, len(turns)+1, len(b.m.keys()))

        c = b.animals_decision(cur)
        if cur.alive:
            turns.append(cur)
        if c is not None:
            #print("child", c.id)
            turns.append(c)
        turns = [a for a in turns if a.alive]
        # for a in turns:
        #     assert a in b.animals

        if (cur.species_id, cur.id) not in saved_networks:
            saved_networks[(cur.species_id, cur.id)] = cur.network

        v = 0.
        for a in turns:
            v += float(np.mean(a.sound_sense))
        if len(turns) != 0:
            v /= len(turns)
        avg_sens.append(v)

        v = []

        v = 0.
        v2 = []
        cc = 0.
        for a in turns:
            if a.sound is not None:
                v += float(a.sound)
                v2.append(float(a.sound))
                cc += 1
        if cc > 0:
            v /= cc
        else:
            v = np.nan
        avg_sound.append(v)
        std_sound.append(np.std(v2))

        v = 0.
        cc = 0.
        for a in turns:
                v += float(a.hp)
                cc += 1
        # if cc > 0:
        #     v /= cc
        # else:
        #     v = np.nan
        avg_hp.append(v)

    plotter()




