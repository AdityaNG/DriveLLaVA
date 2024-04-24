"""
Models
"""

import numpy as np

from drivellava.common import DEG_2_RAD, KMPH_2_MPS

from .image import Image
from .packet import Packet


class DroneControls(Packet):
    """Controls Request"""

    trajectory_index: int
    speed_index: int


class DroneState(Packet):
    """Drone State"""

    image: Image  # jpeg encoded image
    velocity_x: float
    velocity_y: float
    velocity_z: float
    omega_x: float
    omega_y: float
    omega_z: float
    steering_angle: float

    def is_stationary(self) -> bool:
        speed = np.sqrt(
            self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2
        )
        if speed > 5 * KMPH_2_MPS:
            return False

        angular_speed = np.sqrt(
            self.omega_x**2 + self.omega_y**2 + self.omega_z**2
        )

        if angular_speed > 30 * DEG_2_RAD:
            return False

        return True
