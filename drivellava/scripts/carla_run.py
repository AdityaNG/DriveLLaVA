"""
GPT Vision to make Control Decisions
"""

import time
import traceback

import cv2
import numpy as np
from matplotlib import colormaps

from drivellava.carla.client import CarlaClient
from drivellava.gpt.gpt_vision import GPTVision
from drivellava.settings import settings
from drivellava.utils import plot_steering_traj


def main():  # pragma: no cover
    """
    Use the Image from the drone and the SLAM map to feed as input to GPT
    Return the drone controls
    """

    mission = settings.system.SYSTEM_MISSION

    print("mission", mission)

    gpt = GPTVision()
    client = CarlaClient()

    trajectory_templates = gpt.trajectory_encoder.left_to_right_traj()
    NUM_TEMLATES = gpt.num_trajectory_templates

    # Select colors based on templates
    COLORS = [
        (255 * colormaps["gist_rainbow"]([float(i + 1) / NUM_TEMLATES])[0])
        for i in range(NUM_TEMLATES)
    ]

    drone_state = client.get_car_state()
    # image = drone_state.image.cv_image()

    last_update = drone_state.timestamp

    try:
        while True:
            client.game_loop()

            drone_state = client.get_car_state(default=drone_state)
            image_raw = np.array(
                drone_state.image.cv_image(),
            )
            image = np.array(
                drone_state.image.cv_image(),
            )

            # Draw all template trajectories
            for index in range(NUM_TEMLATES):
                template_trajectory = trajectory_templates[index]
                template_trajectory_3d = np.zeros(
                    (settings.system.TRAJECTORY_SIZE, 3)
                )
                template_trajectory_3d[:, 0] = template_trajectory[:, 0]
                template_trajectory_3d[:, 2] = template_trajectory[:, 1]
                color = COLORS[index]
                plot_steering_traj(
                    image, template_trajectory_3d, color=color, track=False
                )

            # print(type(image), image.dtype)

            image_vis = image_raw.copy()
            # Get GPT Controls
            if drone_state.timestamp > last_update:
                # last_update = drone_state.timestamp
                last_update = int(time.time() * 1000)
                gpt_controls = gpt.step(image, drone_state, mission)

                client.set_car_controls(gpt_controls, gpt.trajectory_encoder)
                gpt.previous_messages.timestamp = last_update

                template_trajectory = trajectory_templates[
                    gpt_controls.trajectory_index
                ]
                template_trajectory_3d = np.zeros(
                    (settings.system.TRAJECTORY_SIZE, 3)
                )
                template_trajectory_3d[:, 0] = template_trajectory[:, 0]
                template_trajectory_3d[:, 2] = template_trajectory[:, 1]

                plot_steering_traj(
                    image_vis,
                    template_trajectory_3d,
                    color=(255, 0, 0),
                    track=True,
                )

                cv2.imshow("carla", image_vis)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        print("Land drone on keyboard interrupt, exiting...")
    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
