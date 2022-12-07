import pygame as pg
import numpy as np

class Road:
    def __init__(self, pos, size, ori, br=None):
        if br is None:
            br = [0, 0, 0, 0]
        self.pos = pos
        self.size = size
        self.br = br
        self.ori = ori
        self.rect = pg.Rect([self.pos, self.size])

    def draw(self, win):
        pg.draw.rect(win, "grey", [self.pos, self.size], border_top_left_radius=self.br[0],
                                                         border_top_right_radius=self.br[1],
                                                         border_bottom_right_radius=self.br[2],
                                                         border_bottom_left_radius=self.br[3])

    def get_checkpoints(self):
        if self.ori == 1:
            return list([x, 1] for x in np.linspace(self.rect.left + 100, self.rect.right - 100, self.rect.width // 5 + 1))
        if self.ori == 2:
            return list([x, 2] for x in np.linspace(self.rect.bottom - 100, self.rect.top + 100, self.rect.height // 5 + 1))
        if self.ori == 3:
            return list([x, 3] for x in np.linspace(self.rect.right - 100, self.rect.left + 100, self.rect.width // 5 + 1))
        if self.ori == 4:
            return list([x, 4] for x in np.linspace(self.rect.top + 100, self.rect.bottom - 100, self.rect.height // 5 + 1))
