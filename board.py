import tkinter as tk
import numpy as np
import animal


shouldDraw = True
canvas_width = 1600
canvas_height = 800
if shouldDraw:
    master = tk.Tk()
sound_vol = 20
linear_falloff = 1
age_loss_bound = 50
move_sound_angle = []
n_species = 1
species_color_dict = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white"]

def v_length(x, y):
    return np.sqrt(x*x + y*y)


def angle_between_vectors(x1, y1, x2, y2):
    dot = x1 * x2 + y1 * y2
    cosa = dot / (v_length(x1, y1)*v_length(x2, y2))
    return np.abs(np.degrees(np.arccos(cosa)) - 180) % 360


def checkered(canvas, h, b):
    # vertical lines at an interval of "line_distance" pixel
    line_distance_h = int(canvas_height/h)
    for x in range(line_distance_h, canvas_width, line_distance_h):
        canvas.create_line(x, 0, x, canvas_height, fill="#476042")
    # horizontal lines at an interval of "line_distance" pixel
    line_distance_b = int(canvas_width/b)
    for y in range(line_distance_b, canvas_height, line_distance_b):
        canvas.create_line(0, y, canvas_width, y, fill="#476042")


def get_box_boundries(x, y, h, b):
    tly = int(float(y * canvas_height)/h) + 1
    tlx = int(float(x * canvas_width)/b) + 1
    bry = int(float((y+1.) * canvas_height)/h) - 1
    brx = int(float((x+1.) * canvas_width)/b) - 1
    return (tlx, tly), (brx, bry)


def adj(x, y):
    return [(x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1), (x - 1, y), (x - 1, y - 1), (x, y - 1),
            (x + 1, y - 1)]


class Board:
    def __init__(self, h, b):
        self.m = {}
        self.animals = []
        self.h, self.b = h, b
        if shouldDraw:
            self.w = tk.Canvas(master, width=canvas_width, height=canvas_height)
            self.w.pack()
            checkered(self.w, h, b)

    def draw_animal(self, a):
        if shouldDraw:
            for o in a.obj:
                self.w.delete(o)
            a.obj = []
            assert a.alive
            (tlx, tly), (brx, bry) = get_box_boundries(a.x, a.y, self.h, self.b)

            fill = species_color_dict[a.species_id]
            si = a.prev_si
            wd = 1 if a.sound is None else 4

            if a.hp <= 0:
                pass

            else:
                a.obj.append(self.w.create_oval(tlx, tly, brx, bry, fill=fill, width=wd))
                if si is not None:
                    ssi = np.sum(si, axis=1)
                    cx = int(tlx + brx)//2
                    cy = int(tly + bry)//2
                    # a.obj.append(self.w.create_line(brx, cy, brx+2*ssi[0]+1, cy, fill="#476042", width=2))
                    # a.obj.append(self.w.create_line(tlx, cy, tlx-2*ssi[2]-1, cy, fill="#476042", width=2))
                    # a.obj.append(self.w.create_line(cx, bry, cx, bry+2*ssi[1]+1, fill="#476042", width=2))
                    # a.obj.append(self.w.create_line(cx, tly, cx, tly-2*ssi[3]-1, fill="#476042", width=2))

    def undraw_animal(self, a):
        if shouldDraw:
            for o in a.obj:
                self.w.delete(o)
            a.obj = []

    def lay_start(self, num):
        pos_list = []
        for x in range(self.b):
            for y in range(self.h):
                pos_list.append((x, y))
        np.random.shuffle(pos_list)
        for n in range(num):
            x, y = pos_list[n]
            # tid = np.random.randint(0, 4)
            # if tid > 1:
            #     tid = 0
            # tid = np.random.randint(0, n_species)
            tid = 0
            a = animal.Animal(x, y, s_id=tid, pred=tid)
            self.m[(x, y)] = a
            self.animals.append(a)
            self.draw_animal(a)

    def check_placable(self, x, y):
        if x < 0 or x >= self.b or y < 0 or y >= self.h:
            return False
        return (x, y) not in self.m.keys()

    def sound_input(self, a):
        si = np.zeros((4, animal.freq_num))
        for t in self.animals:
            if t.alive and t != a:
                if t.sound is not None:
                    distance = np.sqrt(np.square(t.x - a.x) + np.square(t.y - a.y))
                    vol = sound_vol - distance * linear_falloff
                    if vol > a.sound_sense[t.sound]:
                        x_comp = vol * float(t.x - a.x) / distance
                        y_comp = vol * float(t.y - a.y) / distance
                        if x_comp > 0:
                            si[0, t.sound] = max(x_comp, si[0, t.sound])
                        else:
                            si[2, t.sound] = max(-x_comp, si[2, t.sound])
                        if y_comp > 0:
                            si[1, t.sound] = max(y_comp, si[1, t.sound])
                        else:
                            si[3, t.sound] = max(-y_comp, si[3, t.sound])

        return si

    def animals_decision(self, a):
        net_hp = sum([lps.hp for lps in self.animals if lps.alive])
        if a.hp <= 0:
            a.alive = False
            self.undraw_animal(a)
            self.m.pop((a.x, a.y))
            return None
        assert a.alive
        a.age += 1
        a.true_age += 1

        if a.true_age % (age_loss_bound * 2) == 0 and a.true_age != 0:
            a.hp -= 50
            tsa = [alt for alt in self.animals if alt.alive and alt.species_id == a.species_id]
            np.random.shuffle(tsa)

            for alt in tsa:
                if alt.true_age < a.true_age:
                    alt.hp += 50
                    break
        x, y = a.x, a.y

        if a.predator:
            a.hp -= 1.

        si = self.sound_input(a)
        si += (np.random.random(size=si.shape))/7.5
        a.prev_si = si

        rot_si = np.argmax(np.sum(si, axis=1))
        assert 0 <= rot_si < 4
        rot_lst = np.arange(0, 4)
        assert rot_lst.shape[0] == 4
        rot_lst_in = np.mod((rot_lst + 4 - rot_si), 4)
        rot_lst_out = np.mod((rot_lst + rot_si), 4)
        rotated_si = np.zeros(si.shape)
        for i in range(rotated_si.shape[0]):
            rotated_si[rot_lst_in[i], :] = si[i, :]
        # print(rot_si, si, rotated_si)

        # outs = a.make_decision(si)
        # assert rotated_si[0, 0] == np.max(rotated_si)
        outs = a.make_decision(rotated_si)

        rotated_move_outs = outs[:4]
        # move_outs = outs[:4]

        move_outs = np.zeros(rotated_move_outs.shape)
        for i in range(move_outs.shape[0]):
            move_outs[rot_lst_out[i]] = rotated_move_outs[i]
        # print(rot_si, rotated_move_outs, move_outs)

        sound_outs = outs[4:]
        if move_outs[0] > 0.5 and move_outs[2] > 0.5:
            pass
        elif move_outs[0] > 0.5:
            if self.check_placable(x+1, y):
                x += 1
        elif move_outs[2] > 0.5:
            if self.check_placable(x-1, y):
                x -= 1
        if move_outs[1] > 0.5 and move_outs[3] > 0.5:
            pass
        if move_outs[1] > 0.5:
            if self.check_placable(x, y+1):
                y += 1
        if move_outs[3] > 0.5:
            if self.check_placable(x, y-1):
                y -= 1
        self.m.pop((a.x, a.y))
        self.m[(x, y)] = a
        # move_sound_angle.append(angle_between_vectors(
        #     x-a.x, y-a.y, move_outs[0]-move_outs[2], move_outs[3]-move_outs[1]))
        # move_sound_angle.append(move_outs[rot_si] > 0.5 > move_outs[(rot_si+2) % 4])
        move_sound_angle.append(rotated_move_outs)
        a.x, a.y = x, y

        # f = np.argmax(sound_outs)

        # if sound_outs[0] > 0*1. / 2:
        a.sound = a.species_id
        # else:
        #     a.sound = None
        self.draw_animal(a)

        near_locations = adj(x, y)
        np.random.shuffle(near_locations)
        made = False
        c = None
        # proximity_loss = sum([1 for (tx, ty) in near_locations if (tx, ty) in self.m.keys()])
        # a.hp -= proximity_loss
        for (lx, ly) in near_locations:
            if made:
                break
            if (lx, ly) in self.m.keys():
                a2 = self.m[(lx, ly)]
                plausible = [v for v in near_locations]
                plausible.extend(adj(lx, ly))
                np.random.shuffle(plausible)
                for (bx, by) in plausible:

                    # random birth location
                    bx = np.random.randint(0, self.b)
                    by = np.random.randint(0, self.h)

                    if self.check_placable(bx, by):
                        bred = animal.Animal.breed(a, a2, (bx, by))
                        if bred:
                            c = bred
                            a.hp -= 50
                            a2.hp -= 50
                            a.age = 0
                            a2.age = 0
                            if a2.hp <= 0:
                                a2.alive = False
                                self.m.pop((a2.x, a2.y))
                                self.undraw_animal(a2)
                            made = True
                            self.m[(bx, by)] = c
                            self.animals.append(c)
                            self.draw_animal(c)
                            break
                        else:
                            animal.cur_id -= 1

        if a.predator:
            for (lx, ly) in near_locations:
                if (lx, ly) in self.m.keys():
                    a2 = self.m[(lx, ly)]
                    if not a2.predator:
                        a2.hp -= 25
                        a.hp += 25
                        tsa = [alt for alt in self.animals if alt.alive and alt.species_id == a2.species_id]
                        np.random.shuffle(tsa)
                        for alt in tsa:
                            if alt.true_age < a2.true_age:
                                alt.hp += 25
                                break
                        tsa = [alt for alt in self.animals if alt.alive and alt.species_id == a.species_id]
                        np.random.shuffle(tsa)
                        for alt in tsa:
                            if alt.true_age < a.true_age:
                                alt.hp -= 12.5
                                if alt.hp <= 0:
                                    alt.alive = False
                                    self.m.pop((alt.x, alt.y))
                                    self.undraw_animal(alt)
                                break

                        if a2.hp <= 0:
                            a2.alive = False
                            self.m.pop((a2.x, a2.y))
                            self.undraw_animal(a2)
        if a.hp <= 0:
            a.alive = False
            self.m.pop((a.x, a.y))
            self.undraw_animal(a)
        net_hp2 = sum([lps.hp for lps in self.animals if lps.alive])
        #print(net_hp, net_hp2)

        return c

    def clear_animals_residue(self):
        for a in self.animals:
            if a.alive:
                assert a == self.m[(a.x, a.y)]
            else:
                assert a not in self.m.keys()
