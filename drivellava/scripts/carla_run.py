"""
GPT Vision to make Control Decisions
"""

import traceback

import cv2
import numpy as np

from drivellava.carla.client import CarlaClient

# from drivellava.gpt.gpt_vision import GPTVision
from drivellava.settings import settings


def main():  # pragma: no cover
    """
    Use the Image from the drone and the SLAM map to feed as input to GPT
    Return the drone controls
    """

    mission = settings.system.SYSTEM_MISSION

    print("mission", mission)

    # gpt = GPTVision()
    client = CarlaClient()

    drone_state = client.get_car_state()
    # image = drone_state.image.cv_image()

    # last_update = drone_state.timestamp

    try:
        while True:
            client.game_loop()

            # # Get GPT Controls
            # if (
            #     drone_state.timestamp > last_update
            # ):
            #     # last_update = drone_state.timestamp
            #     last_update = int(time.time() * 1000)
            #     # gpt_controls = gpt.step(image, drone_state, mission)

            #     # client.set_car_controls(gpt_controls)
            #     gpt.previous_messages.timestamp = last_update

            drone_state = client.get_car_state(default=drone_state)
            image = np.array(
                drone_state.image.cv_image(),
            )

            # print(type(image), image.dtype)

            cv2.imshow("carla", image)
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
