import math
from typing import Tuple

import numpy as np
import pygame

class Car:
    def __init__(self, acceleration:float = 0.2, friction:float = 0.1, max_speed:float = 7, min_rotation_speed:float = 2, max_rotation_speed:float = 4, rotation:float = 90) -> None:
        self.speed = 0
        self.acceleration = acceleration
        self.friction = friction
        self.max_speed = max_speed
        self.min_rotation_speed = min_rotation_speed
        self.max_rotation_speed = max_rotation_speed
        self.rotation_speed_range = self.max_rotation_speed - self.min_rotation_speed
        self.rotation = rotation

        self.image = pygame.image.load("images/car.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (50, 25))
        
        self.rect = self.image.get_rect()

        self.image_rotated = pygame.transform.rotate(self.image, 0)
        self.rect_rotated = self.image_rotated.get_rect(center=self.rect.center)

    def set_position(self, position:Tuple[int, int]) -> None:
        self.rect.center = position

    def draw(self, screen:pygame.display) -> None:
        screen.blit(self.image_rotated, self.rect_rotated)

    def move(self, action:Tuple[int, int]) -> None:
        speed, rotation = action

        if abs(self.speed) < self.max_speed:
            self.speed += speed * self.acceleration

        if self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction

        if abs(self.speed) > 1:
            self.rotation += rotation * (self.min_rotation_speed + (self.rotation_speed_range * (1 - abs(self.speed) / self.max_speed)))

        angle_rad = math.radians(-self.rotation+90)
        velocity_x = math.sin(angle_rad) * self.speed
        velocity_y = math.cos(angle_rad) * self.speed

        self.rect.x += velocity_x
        self.rect.y -= velocity_y  # Subtract velocity_y to move upwards (opposite to the y-axis direction)

        self.image_rotated = pygame.transform.rotate(self.image, self.rotation)
        self.rect_rotated = self.image_rotated.get_rect(center=self.rect.center)

    def keep_inbounds(self, screen_width:int, screen_height:int) -> None:
        # Keep the car from going off the screen
        if self.rect.left < 0:
            self.rect.left = 0
        elif self.rect.right > screen_width:
            self.rect.right = screen_width
        if self.rect.top < 0:
            self.rect.top = 0
        elif self.rect.bottom > screen_height:
            self.rect.bottom = screen_height
