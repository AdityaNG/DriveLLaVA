import cv2
import numpy as np

try:
    import pygame
except ImportError:
    raise RuntimeError(
        "cannot import pygame, make sure pygame package is installed"
    )

import carla
from PIL import Image as PILImage

from drivellava.carla.helpers import HUD, KeyboardControl, World
from drivellava.schema.carla import DroneControls, DroneState
from drivellava.schema.image import Image
from drivellava.trajectory_encoder import TrajectoryEncoder


class CarlaClient:

    def __init__(
        self,
        host="127.0.0.1",
        port=2000,
        sync=True,
        autopilot=False,
        width=256,
        height=128,
        rolename="hero",
        filter="vehicle.tesla.model3",
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
            # if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0
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
        # img = np.zeros((256, 256, 3), dtype=np.uint8)
        data = pygame.image.tostring(self.display, "RGBA")
        W, H = self.display.get_width(), self.display.get_height()
        img_pil = PILImage.frombytes("RGBA", (W, H), data)
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = Image(data=img.tolist())

        vel = self.world.player.get_velocity()
        controls = self.world.player.get_control()

        return DroneState(
            image=image,
            velocity_x=vel.x,
            velocity_y=vel.y,
            velocity_z=vel.z,
            steering_angle=controls.steer,
        )

    def set_car_controls(
        self, controls: DroneControls, trajectory_encoder: TrajectoryEncoder
    ):
        """
        Set the car controls
        """
        controls.speed_index = 1
        if controls.speed_index == 0:
            self.world.player.enable_constant_velocity(
                carla.Vector3D(0, 0, 0)  # 0 Km/h
            )
            self.world.constant_velocity_enabled = True
        elif controls.speed_index == 1:
            self.world.player.enable_constant_velocity(
                carla.Vector3D(8.33, 0, 0)  # 30 Km/h
            )
            self.world.constant_velocity_enabled = True

            steering_angle = (
                2.0
                * (
                    float(controls.trajectory_index)
                    / trajectory_encoder.num_trajectory_templates
                )
                - 1.0
            )

            assert -1 <= steering_angle <= 1

            self.controller.set_steering_angle(steering_angle)
