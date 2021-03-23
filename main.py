import board
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time
import animal


take_inputs = False


def plotter():
    temp = [np.mean(board.move_sound_angle[max(i-50, 0): min(i+50, len(board.move_sound_angle)-1)])
            for i in range(len(board.move_sound_angle))]
    plt.plot(temp)

    plt.title('angle')

    plt.figure()
    plt.plot(avg_sens)
    plt.title('sens')

    plt.figure()
    plt.plot(avg_sound, color='blue')
    plt.plot(std_sound, color='red')
    plt.title("avg sound freq, std")

    plt.figure()
    plt.plot(avg_hp, color='black')
    plt.title('hp')

    plt.figure()
    plt.plot(avg_sound_genes, color='blue')
    plt.plot(std_sound_genes, color='red')
    plt.title('sound genes')

    plt.figure()
    sensitive = [a.sound_sense for a in turns]
    sss = np.zeros(sensitive[0].shape)
    for ss in sensitive:
        sss += ss
    print(sss)
    sss /= len(sensitive)
    ms = [a.sound for a in turns if a.sound is not None]
    plt.hist(ms, bins=animal.freq_num, range=(-0.5, animal.freq_num+0.5))
    plt.plot(sss)
    plt.title('avg sensitivity')

    plt.show()


if __name__ == "__main__":
    num = 30
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

    while len(turns) > 0:
        if count % (num) == 0:
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
        print(count, cur.id, len(turns)+1, len(b.m.keys()))

        c = b.animals_decision(cur)
        turns.append(cur)
        if c is not None:
            print("child", c.id)
            turns.append(c)
        turns = [a for a in turns if a.alive]
        # for a in turns:
        #     assert a in b.animals

        v = 0.
        for a in turns:
            v += float(np.mean(a.sound_sense))
        if len(turns) != 0:
            v /= len(turns)
        avg_sens.append(v)

        v = []
        for a in turns:
            v.append(float(np.mean(a.genes[4:, :])))
        avg_sound_genes.append(np.mean(v))
        std_sound_genes.append(np.std(v))

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




