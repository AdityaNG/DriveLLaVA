"""
GPT Vision to make Control Decisions
"""

import os
import signal
import subprocess
import textwrap
import time
import traceback
from contextlib import contextmanager

import cv2
import numpy as np
from torch.multiprocessing import Process, Queue

from drivellava.carla.client import CarlaClient
from drivellava.gpt.gpt_vision import GPTVision
from drivellava.schema.carla import DroneControls
from drivellava.settings import settings
from drivellava.utils import plot_steering_traj


def async_gpt(gpt_input_q: Queue, gpt_output_q: Queue, gpt: GPTVision):

    while True:
        try:
            data = gpt_input_q.get()
            image, drone_state, mission = data
            gpt_controls = gpt.step(image, drone_state, mission)
            prompt_str = gpt.previous_messages.to_str()
            gpt_output_q.put((gpt_controls.model_dump(), prompt_str))
        except Exception as ex:
            print("Exception while calling GPT", ex)
            traceback.print_exc()


@contextmanager
def start_carla():
    """
    Launch Carla using the command

    This function is to be used with the 'with' clause:
        ```py
        with start_carla():
            print("Carla is running")
        ```

        Once the function exits, carla is to shutdown
    """
    command = "CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh -quality-level=Low -prefernvidia -ResX=10 -ResY=10"  # noqa
    try:
        # Start the Carla process
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=settings.ui.CARLA_INSTALL_PATH,
            preexec_fn=os.setsid,
        )
        time.sleep(5)
        yield process
    finally:
        # Ensure the Carla process is terminated upon exiting the context
        # Note: it appears that carla requires two of these to exit
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(1)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        process.wait()  # Wait for the process to properly terminate


def print_text_image(
    img,
    text,
    width=50,
    font_size=0.5,
    font_thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    wrapped_text = textwrap.wrap(text, width=width)

    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 10

        y = (i + 1) * gap
        x = 10

        cv2.putText(
            img,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def main():  # pragma: no cover
    """
    Use the Image from the sim and the map to feed as input to GPT
    Return the vehicle controls to the sim and step
    """

    mission = settings.system.SYSTEM_MISSION

    print("mission", mission)

    gpt = GPTVision()
    client = CarlaClient()

    trajectory_templates, colors = (
        gpt.trajectory_encoder.get_colors_left_to_right()
    )
    NUM_TEMLATES = gpt.num_trajectory_templates

    drone_state = client.get_car_state()
    # image = drone_state.image.cv_image()

    last_update = drone_state.timestamp

    gpt_input_q = Queue(maxsize=1)
    gpt_output_q = Queue(maxsize=1)

    gpt_process = Process(
        target=async_gpt, args=(gpt_input_q, gpt_output_q, gpt)
    )
    gpt_process.daemon = True
    gpt_process.start()

    try:
        while True:
            visual = np.zeros((512, 512, 3), dtype=np.uint8)
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
                color = colors[index]
                plot_steering_traj(
                    image, template_trajectory_3d, color=color, track=False
                )

            print_text_image(image, "Prompt")
            visual[0:128, 0:256] = image

            image_vis = image_raw.copy()

            if gpt_input_q.empty():
                data = (image, drone_state, mission)
                gpt_input_q.put(data)

            # Get GPT Controls
            if not gpt_output_q.empty() or (
                settings.system.GPT_WAIT
                and time.time() * 1000 - last_update > 10.0 * 1000
            ):
                gpt_controls_dict, prompt_str = gpt_output_q.get()
                gpt_controls = DroneControls(**gpt_controls_dict)

                client.set_car_controls(gpt_controls, gpt.trajectory_encoder)
                gpt.previous_messages.timestamp = last_update
                last_update = gpt_controls.timestamp

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
                print_text_image(image_vis, "DriveLLaVA Controls")
                visual[0:128, 256:512] = image_vis

                text_visual = np.zeros((512 - 128, 512, 3), dtype=np.uint8)
                print_text_image(
                    text_visual,
                    prompt_str,
                    width=80,
                    font_size=0.35,
                    font_thickness=1,
                )
                visual[128:512, 0:512] = text_visual

            cv2.imshow("DriveLLaVA UI", visual)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        print("Land drone on keyboard interrupt, exiting...")
    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        gpt_process.kill()


if __name__ == "__main__":
    with start_carla():
        main()
