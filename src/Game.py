import gym
import math
import numpy as np
import pygame
from typing import Tuple, List
from Car import Car
from Checkpoint import Checkpoint
from Sensor import Sensor

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

class Game(gym.Env):
    def __init__(self, screen_width:int = 1200, screen_height:int = 800) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height

    def init(self) -> None:
        self.car = Car()
        self.car.set_position((110, 500))

        self.sensor_info = []
        self.sensors = [
            Sensor(500, pygame.math.Vector2(1, 0)),
            Sensor(500, pygame.math.Vector2(math.cos(-math.pi/7), math.sin(-math.pi/7))),
            Sensor(500, pygame.math.Vector2(math.cos(math.pi/7), math.sin(math.pi/7))),
            Sensor(500, pygame.math.Vector2(math.cos(-math.pi/3), math.sin(-math.pi/3))),
            Sensor(500, pygame.math.Vector2(math.cos(math.pi/3), math.sin(math.pi/3))),
            Sensor(500, pygame.math.Vector2(math.cos(-math.pi/2), math.sin(-math.pi/2))),
            Sensor(500, pygame.math.Vector2(math.cos(math.pi/2), math.sin(math.pi/2))),
            Sensor(500, pygame.math.Vector2(math.cos(-3*math.pi/4), math.sin(-3*math.pi/4))),
            Sensor(500, pygame.math.Vector2(math.cos(3*math.pi/4), math.sin(3*math.pi/4))),
            Sensor(500, pygame.math.Vector2(-1, 0)),
        ]

        self.edge_distances = {i: 500 for i in range(len(self.sensors))}

        self.checkpoint_id = 0
        self.checkpoints = [
            Checkpoint(40, 450, 170, 450),
            Checkpoint(60, 370, 195, 370),
            Checkpoint(100, 280, 230, 290),
            Checkpoint(150, 190, 270, 220),
            Checkpoint(230, 105, 310, 170),
            Checkpoint(330, 70, 390, 140),

            Checkpoint(460, 45, 470, 125),
            Checkpoint(750, 35, 730, 115),
            Checkpoint(870, 200, 965, 255),
            Checkpoint(700, 300, 795, 350),
            Checkpoint(820, 490, 875, 415),
            Checkpoint(990, 550, 1090, 505),
            Checkpoint(910, 650, 990, 710),
            Checkpoint(670, 680, 700, 780),
            Checkpoint(400, 650, 385, 730),
            Checkpoint(80, 665, 180, 610)
        ]
    
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

    def step(self, action:Tuple[int, int]) -> List:
        reward = -0.2 if self.car.speed <= 0 else 0.1
        done = False


        self.car.move(action)
        self.car.keep_inbounds(self.screen_width, self.screen_height)
        
        # collision_distances = {id: None for id in range(len(self.sensors))}

        self.sensor_info = []
        for i, sensor in enumerate(self.sensors):
            sensor.move(self.car.rotation, self.car.rect)
            self.edge_distances[i], sensor_inf  = sensor.collisions(self.background_image, self.car.rect)
            self.sensor_info.append(sensor_inf)
        # print(self.edge_distances)
        for collision_distance in self.edge_distances.values():
            if collision_distance is not None and collision_distance < min(self.car.rect.width // 2, self.car.rect.height // 2):
                # self.reset()
                reward = -20
                done = True


        # print(LINE_UP, end=LINE_CLEAR)
        # print(LINE_UP, end=LINE_CLEAR)
        # print(f"Position: {self.car.rect}, Speed: {round(self.car.speed)}, Rotation: {round(self.car.rotation)}")
        # print(f"Collision: {self.edge_distances}")
    
        cleared = False
        # for checkpoint in self.checkpoints:
        if self.checkpoint_id == len(self.checkpoints):
            cleared = True
        else:
            if self.checkpoints[self.checkpoint_id].active == 1 and self.checkpoints[self.checkpoint_id].intersect(self.car.rect_rotated):
            #     self.checkpoints[self.checkpoint_id].active = 0
            # elif self.checkpoints[self.checkpoint_id].active == 0:
                self.checkpoints[self.checkpoint_id].active = -1
                self.checkpoint_id += 1
                reward += 10
            
        if cleared:
            # print("Collected all checkpoints!!")
            self.reset_checkpoints()
            self.checkpoint_id = 0
            reward += 100
            done = True

        # return [np.array([self.car.get_position(), self.car.speed, self.car.rotation, self.edge_distances]), reward, False, None, None]
        x, y = self.car.get_position()
        observation = [x/self.screen_width, y/self.screen_height, self.car.speed/self.car.max_speed, self.car.rotation/360.0]
        for val in self.edge_distances.values():
            observation.append(val/500)

        return [np.array(observation), reward, done, None, None]

    def render(self) -> None:
        self.screen.blit(self.background_image, (0, 0))
        self.car.draw(self.screen)

        for sensor in self.sensors:
            sensor.draw(self.screen, self.car.rect)

        for sensor_inf in self.sensor_info:
            if sensor_inf:
                sensor.draw_point(self.screen, sensor_inf[1], sensor_inf[0], sensor_inf[2])
    
        for i, checkpoint in enumerate(self.checkpoints):
            if checkpoint.active == 1:
                # if i != self.checkpoint_id:
                    # continue
                checkpoint.color = (0, 0, 255)
            # elif checkpoint.active == 0:
            #     checkpoint.color = (255, 0, 0)    # Intersecting checkpoint
            else:
                checkpoint.color = (0, 255, 0)    # Cleared checkpoint
            checkpoint.draw(self.screen)

        pygame.display.update()

    def start(self) -> None:
        while True:
            # self.clock.tick(60)
            self.check_quit()
            keys = pygame.key.get_pressed()
            action = self.get_action(keys)
            self.step(action)
            self.render()
                            
    def check_quit(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def reset_checkpoints(self) -> None:
        self.checkpoint_id = 0
        for checkpoint in self.checkpoints:
            checkpoint.reset()

    def reset_snesors(self) -> None:
        self.sensor_info = []
        for sensor in self.sensors:
            sensor.reset()
    
    def reset_car(self) -> None:
        self.car.set_position((110, 500))
        self.car.speed = 0
        self.car.rotation = 90
    
    def reset_edge_distances(self) -> None:
        for key in self.edge_distances:
            self.edge_distances[key] = 500

    def reset(self) -> List:
        self.reset_car()
        self.reset_snesors()
        self.reset_checkpoints()
        self.reset_edge_distances()
        
        x, y = self.car.get_position()
        observation = [x/self.screen_width, y/self.screen_height, self.car.speed/self.car.max_speed, self.car.rotation/360.0]
        for val in self.edge_distances.values():
            observation.append(val/500)

        return np.array(observation)
    
if __name__ == "__main__":
    screen_width = 1200
    screen_height = 800

    game = Game(screen_width, screen_height)
    game.init_render()
    game.clock.tick(60)
    game.init()
    game.start()
