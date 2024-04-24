import numpy as np

try:
    import pygame
except ImportError:
    raise RuntimeError(
        "cannot import pygame, make sure pygame package is installed"
    )

import carla

from drivellava.carla.helpers import HUD, KeyboardControl, World
from drivellava.schema.carla import DroneControls, DroneState


class CarlaClient:

    def __init__(
        self,
        host="127.0.0.1",
        port=2000,
        sync=False,
        autopilot=False,
        width=1280,
        height=720,
        rolename="hero",
        filter="vehicle.*",
        generation="2",
        gamma=2.2,
    ):
        pygame.init()
        pygame.font.init()
        self.world = None
        self.original_settings = None

        self.client = carla.Client(host, port)
        self.client.set_timeout(2000.0)

        self.sim_world = self.client.get_world()
        if sync:
            settings = self.sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            self.sim_world.apply_settings(settings)

            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if autopilot and not self.sim_world.get_settings().synchronous_mode:
            print(
                "WARNING: You are currently in asynchronous mode and could "
                "experience some issues with the traffic simulation"
            )

        self.display = pygame.display.set_mode(
            (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.display.fill((0, 0, 0))
        pygame.display.flip()

        self.hud = HUD(width, height)
        self.world = World(
            self.sim_world,
            self.hud,
            sync,
            rolename,
            filter,
            generation,
            gamma,
        )
        self.controller = KeyboardControl(self.world, autopilot)

        if sync:
            self.sim_world.tick()
        else:
            self.sim_world.wait_for_tick()

        self.clock = pygame.time.Clock()
        self.sync = sync

    def game_loop(self) -> bool:
        if self.sync:
            self.sim_world.tick()
        self.clock.tick_busy_loop(60)
        if self.controller.parse_events(
            self.client, self.world, self.clock, self.sync
        ):
            # Exit
            return False
        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()
        return True

    def get_car_state(self, default=None) -> DroneState:
        """
        Get the current state of the car
        """
        img = np.array((256, 256, 3), dtype=np.uint8)
        return DroneState(
            image=img.tolist(),
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            omega_x=0.0,
            omega_y=0.0,
            omega_z=0.0,
            steering_angle=0.0,
        )

    def set_car_controls(self, controls: DroneControls):
        """
        Set the car controls
        """
        pass
