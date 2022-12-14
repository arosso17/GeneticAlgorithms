from NeuralNet import *
from Road import *

car = pg.image.load("car_top.png")
car = pg.transform.rotozoom(car, 180, 0.25)

def crossover(c1, c2, checkpoints):
    c1 = Car([100, 500], c1.nn.weights[:], c1.nn.biases[:], checkpoints[:])
    c2 = Car([100, 500], c2.nn.weights[:], c2.nn.biases[:], checkpoints[:])
    c3w = []
    c4w = []
    for par1, par2 in zip(c1.nn.weights, c2.nn.weights):
        crosspoint = int(np.random.rand() * len(c1.nn.weights))
        t3 = []
        t4 = []
        for p in par1[0:crosspoint]:
            t3.append(p)
        for p in par2[crosspoint:]:
            t3.append(p)
        for p in par2[0:crosspoint]:
            t4.append(p)
        for p in par2[crosspoint:]:
            t4.append(p)
        c3w.append(t3)
        c4w.append(t4)
    c3 = Car([100, 500], c3w, c1.nn.biases[:], checkpoints[:])
    c4 = Car([100, 500], c4w, c1.nn.biases[:], checkpoints[:])
    return [c1, c2, c3, c4]

class Car:
    def __init__(self, pos, weights, biases, cp):
        self.nn = NeuralNetwork([6, 5, 4], weights, biases)
        self.cp = cp
        self.score = 0
        self.pos = pos
        self.speed = 0
        self.car_dir = 0
        self.size = car.get_size()
        self.size = [self.size[0] - 10, self.size[1] - 14]
        self.surface = car
        self.angle = 90
        dx = self.size[0] / 2 * np.cos(np.radians(self.angle))
        dy = self.size[0] / 2 * np.sin(np.radians(self.angle))
        self.rotated_image = pg.transform.rotate(self.surface, self.angle)
        self.rect = self.rotated_image.get_rect(center=[self.pos[0] + dx, self.pos[1] - dy])
        self.hp = 5
        self.max_speed = 7.5
        self.crashed = False
        self.drift = False
        self.points = [self.rect.topleft, self.rect.topright, self.rect.bottomright, self.rect.bottomleft]
        self.angles = [-90, -45, 0, 45, 90]
        self.dists = [0, 0, 0, 0, 0]
        self.time = 0

    def draw(self, win, lines, draw, overide=False):
        if not self.crashed:
            cos = np.cos(np.radians(self.angle))
            sin = np.sin(np.radians(self.angle))
            dx1 = self.size[0] / 2 * cos
            dx2 = self.size[1] / 2 * sin
            dy1 = self.size[0] / 2 * sin
            dy2 = self.size[1] / 2 * cos
            self.rotated_image = pg.transform.rotate(self.surface, self.angle)
            # print([self.pos[0] + dx1, self.pos[1] - dy1])
            self.rect = self.rotated_image.get_rect(center=self.pos)

            self.points = [[self.rect.centerx - dx1 - dx2, self.rect.centery + dy1 - dy2],
                           [self.rect.centerx + dx1 - dx2, self.rect.centery - dy1 - dy2],
                           [self.rect.centerx + dx1 + dx2, self.rect.centery - dy1 + dy2],
                           [self.rect.centerx - dx1 + dx2, self.rect.centery + dy1 + dy2]]
        if draw and (not self.crashed or overide):
            win.blit(self.rotated_image, self.rect.topleft)
            # pg.draw.polygon(win, "red", self.points, 2)
            # pg.draw.circle(win, 'blue', self.rect.center, 5)
            if lines:
                for i in range(len(self.dists)):
                    pg.draw.aaline(win, "red", self.rect.center, (
                    self.rect.centerx + self.dists[i] * np.cos(np.radians(self.angle + self.angles[i])),
                    self.rect.centery + self.dists[i] * np.sin(np.radians(self.angle + self.angles[i]) + np.pi)))

    def see(self, win):
        self.dists = [0, 0, 0, 0, 0]
        for i in range(len(self.angles)):
            cr = False
            sin = np.sin(np.radians(self.angle + self.angles[i]) + np.pi)
            cos = np.cos(np.radians(self.angle + self.angles[i]))
            while not cr:
                self.dists[i] += 5
                dx = self.dists[i] * cos
                dy = self.dists[i] * sin
                try:
                    if win.get_at([int(self.rect.centerx + dx), int(self.rect.centery + dy)]) == (0, 255, 0, 255):
                        cr = True
                except Exception as e:
                    cr = True

    def update(self, win, lines, draw, roads):
        self.time += 1
        self.speed *= 0.95  # friction
        self.score += self.speed
        accel, brake, turn_r, turn_l = self.nn.think(self.dists, self.speed)
        accel = accel[0]
        brake = brake[0]
        turn_r = turn_r[0]
        turn_l = turn_l[0]
        if not self.crashed:
            self.see(win)
        if self.speed < self.max_speed and accel > 0.0:
            self.speed += self.hp * accel
        if brake > 0.0:
            self.speed *= brake
        if self.crashed:
            self.speed = 0

        self.angle %= 360

        if turn_r > 0.0 and self.speed != 0:
            self.angle -= turn_r * 10
        if turn_l > 0.0 and self.speed != 0:
            self.angle += turn_l * 10

        if abs(self.speed) < 0.1:
            self.speed = 0
        if self.speed == 0:
            self.crashed = True
        if self.time > 500 and self.score < 1000:
            self.crashed = True
        self.pos[0] += self.speed * np.cos(np.radians(self.angle))
        self.pos[1] += -self.speed * np.sin(np.radians(self.angle))
        if self.cp:  # this is effectively the fitness evaluation
            if self.cp[0][1] == 1:
                if self.rect.centerx >= self.cp[0][0]:
                    self.score += 100
                    self.cp.pop(0)
            elif self.cp[0][1] == 2:
                if self.rect.centery <= self.cp[0][0]:
                    self.score += 100
                    self.cp.pop(0)
            elif self.cp[0][1] == 3:
                if self.rect.centerx <= self.cp[0][0]:
                    self.score += 100
                    self.cp.pop(0)
            elif self.cp[0][1] == 4:
                if self.rect.centery >= self.cp[0][0]:
                    self.score += 100
                    self.cp.pop(0)
        self.draw(win, lines, draw)
