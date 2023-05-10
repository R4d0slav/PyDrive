import gym
import math
import numpy as np
import pygame
from typing import Tuple, List
from Car import Car
from Checkpoint import Checkpoint
from Sensor import Sensor

# LINE_UP = '\033[1A'
# LINE_CLEAR = '\x1b[2K'

class Game(gym.Env):
    def __init__(self, screen_width:int = 1200, screen_height:int = 800) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.init_render()
        self.init()

    def init(self) -> None:
        self.car = Car()
        self.car.set_position((130, 360))

        self.sensors = [
            Sensor(200, pygame.math.Vector2(1, 0)),
            Sensor(200, pygame.math.Vector2(math.cos(-math.pi/7), math.sin(-math.pi/7))),
            Sensor(200, pygame.math.Vector2(math.cos(math.pi/7), math.sin(math.pi/7))),
            Sensor(200, pygame.math.Vector2(math.cos(-math.pi/3), math.sin(-math.pi/3))),
            Sensor(200, pygame.math.Vector2(math.cos(math.pi/3), math.sin(math.pi/3))),
            Sensor(200, pygame.math.Vector2(math.cos(-math.pi/2), math.sin(-math.pi/2))),
            Sensor(200, pygame.math.Vector2(math.cos(math.pi/2), math.sin(math.pi/2))),
            # Sensor(200, pygame.math.Vector2(math.cos(-3*math.pi/4), math.sin(-3*math.pi/4))),
            # Sensor(200, pygame.math.Vector2(math.cos(3*math.pi/4), math.sin(3*math.pi/4))),
            # Sensor(200, pygame.math.Vector2(-1, 0)),
        ]

        self.checkpoints = [
            Checkpoint(40, 450, 170, 450),
            Checkpoint(100, 280, 235, 280),
            Checkpoint(230, 105, 310, 170),
            Checkpoint(460, 45, 470, 125),
            Checkpoint(750, 35, 730, 115),
            Checkpoint(870, 200, 965, 255),
            Checkpoint(700, 300, 795, 350),
            Checkpoint(820, 490, 875, 415),
            Checkpoint(1000, 560, 1090, 505),
            Checkpoint(910, 650, 990, 710),
            Checkpoint(670, 680, 700, 780),
            Checkpoint(400, 650, 385, 730),
            Checkpoint(80, 665, 180, 610)
        ]

        self.start()
    
    def init_render(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Car Game")
        self.clock = pygame.time.Clock()

        self.background_image = pygame.image.load("images/map.png")
        self.background_image = pygame.transform.scale(self.background_image, (self.screen_width, self.screen_height))

    def get_action(self, keys: Tuple[int, ...]) -> np.ndarray:
        actions = {
            pygame.K_LEFT:  np.array([0, 1]),   # Rotate left
            pygame.K_RIGHT: np.array([0, -1]),  # Rotate right
            pygame.K_UP:    np.array([1, 0]),   # Increase speed
            pygame.K_DOWN:  np.array([-1, 0])   # Decrease speed
        }

        action = np.array([0, 0])
        for key in actions:
            if keys[key]:
                action += actions[key]

        return action

    def step(self, action:Tuple[int, int]) -> Tuple:
        self.car.move(action)
        self.car.keep_inbounds(self.screen_width, self.screen_height)
        
        collision_distances = {id: None for id in range(len(self.sensors))}
        sensor_info = []
        for i, sensor in enumerate(self.sensors):
            sensor.move(self.car.rotation, self.car.rect)
            collision_distances[i], sensor_inf  = sensor.collisions(self.background_image, self.car.rect)
            sensor_info.append(sensor_inf)

        for collision_distance in collision_distances.values():
            if collision_distance is not None and collision_distance < 10:
                self.reset()
        # print(LINE_UP, end=LINE_CLEAR)
        # print(f"Position: {self.car.rect}, Speed: {round(self.car.speed)}, Rotation: {round(self.car.rotation)}, Collision: {collision_distances}")
    
        cleared = True
        checkpoint_intersect = None
        for checkpoint in self.checkpoints:
            if checkpoint.active != -1:
                cleared = False
                if checkpoint.intersect(self.car.rect_rotated):
                    checkpoint_intersect = checkpoint
                    checkpoint.active = 0
                elif checkpoint.active == 0:
                    checkpoint.active = -1
            
        if cleared:
            # print("Collected all checkpoints!!")
            for checkpoint in self.checkpoints:
                checkpoint.active = 1

        return collision_distances, sensor_info, checkpoint_intersect
        # return observation, reward, done, info

    def render(self, sensor_info:List) -> None:
        self.screen.blit(self.background_image, (0, 0))
        self.car.draw(self.screen)

        for sensor in self.sensors:
            sensor.draw(self.screen, self.car.rect)

        for sensor_inf in sensor_info:
            if sensor_inf:
                sensor.draw_point(self.screen, sensor_inf[1], sensor_inf[0], sensor_inf[2])
        
        for checkpoint in self.checkpoints:
            if checkpoint.active == 1:
                checkpoint.color = (0, 0, 255)    # Active checkpoint
            elif checkpoint.active == 0:
                checkpoint.color = (255, 0, 0)    # Intersecting checkpoint
            else:
                checkpoint.color = (0, 255, 0)    # Cleared checkpoint
            checkpoint.draw(self.screen)
        
        pygame.display.update()

    def start(self) -> None:
        while True:
            self.clock.tick(60)
            self.check_quit()
            keys = pygame.key.get_pressed()
            action = self.get_action(keys)
            _, sensor_info, _ = self.step(action)
            self.render(sensor_info)
                            
    def check_quit(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def reset(self) -> None:
        self.init()
        # return observation

if __name__ == "__main__":
    screen_width = 1200
    screen_height = 800

    Game(screen_width, screen_height)