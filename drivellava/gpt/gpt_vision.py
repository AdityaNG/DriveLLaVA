"""
GPT Vision to make Control Decisions
"""

from textwrap import dedent

import instructor
import numpy as np
from openai import OpenAI

from drivellava.gpt.prompts import (
    GPT_PROMPT_CONTROLS,
    GPT_PROMPT_SETUP,
    GPT_PROMPT_UPDATE,
    GPT_SYSTEM,
)
from drivellava.schema.carla import DroneControls, DroneState
from drivellava.schema.gpt import (
    GPTMessage,
    GPTMessageImageContent,
    GPTMessageTextContent,
    GPTState,
)
from drivellava.settings import settings
from drivellava.trajectory_encoder import TrajectoryEncoder

VISION_MODEL = settings.system.SYSTEM_VISION_MODEL
CONTROLS_MODEL = settings.system.SYSTEM_CONTROLS_MODEL

# VISION_MODEL = "llava:latest"
# CONTROLS_MODEL = "mistral:instruct"


def llm_client_factory(llm_provider: str):
    if settings.system.SYSTEM_LLM_PROVIDER == "openai":
        return instructor.patch(OpenAI())

    if settings.system.SYSTEM_LLM_PROVIDER == "ollama":
        return instructor.patch(
            OpenAI(
                base_url="http://localhost:11434/v1/",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )

    raise Exception(f"Unknown LLM provider: {llm_provider}")


class GPTVision:
    def __init__(self, max_history=0):
        self.client = llm_client_factory(settings.system.SYSTEM_LLM_PROVIDER)

        self.previous_messages = GPTState()
        self.max_history = max_history
        self.num_trajectory_templates = 5
        TRAJECTORY_TEMPLATES_NPY = f"./trajectory_templates/proposed_trajectory_templates_simple_{self.num_trajectory_templates}.npy"  # noqa
        TRAJECTORY_TEMPLATES_KMEANS_PKL = f"./trajectory_templates/kmeans_simple_{self.num_trajectory_templates}.pkl"  # noqa
        self.trajectory_encoder = TrajectoryEncoder(
            num_trajectory_templates=self.num_trajectory_templates,
            trajectory_templates_npy=TRAJECTORY_TEMPLATES_NPY,
            trajectory_templates_kmeans_pkl=TRAJECTORY_TEMPLATES_KMEANS_PKL,
        )

    def step(
        self,
        image: np.ndarray,
        state: DroneState,
        mission: str,
    ) -> DroneControls:
        # base64_image = encode_opencv_image(image)

        if not settings.system.GPT_ENABLED:
            gpt_controls = DroneControls(trajectory_index=0, speed_index=0)
            return gpt_controls

        # If number of messages is greater than max_history,
        # remove the oldest message. Do not remove the system message
        # The -2 is to account for the [system message, setup message]
        if len(self.previous_messages) - 2 > self.max_history:
            EXTRA_SIZE = len(self.previous_messages)
            for _ in range(2, EXTRA_SIZE):
                self.previous_messages.pop(-1)

        trajectory_templates, colors = (
            self.trajectory_encoder.get_colors_left_to_right()
        )
        offset = (self.num_trajectory_templates - 1) // 2
        color_map = {i - offset: colors[i] for i in range(len(colors))}
        center = (self.num_trajectory_templates - 1) // 2 - offset
        first = 0 - offset
        last = self.num_trajectory_templates - 1 - offset
        traj_str = dedent(
            f"""
            The trajectories are numbered from:
            [{first},{last}]

            They are labelled from left to right with a BGR color mapping.
            Select trajectory {first} (color: {color_map[first]}) to get the \
            left most.
            Select trajectory {center} (color: {color_map[center]}) to get \
            a centered trajectory
            Select trajectory {last} (color: {color_map[last]}) to get the \
            right most.

            Color Mapping (B,G,R):
            {str(color_map)}
            """
        )
        new_messages = [
            # User
            GPTMessage(
                role="user",
                content=[
                    GPTMessageImageContent(  # type: ignore
                        image=image.tolist(),
                    ),
                    GPTMessageTextContent(  # type: ignore
                        text=GPT_PROMPT_UPDATE.format(
                            mission=mission,
                        ),
                    ),
                ],
            ),
        ]
        if len(self.previous_messages) == 0:
            # System
            system_message = GPTMessage(
                role="system",
                content=[
                    GPTMessageTextContent(  # type: ignore
                        text=GPT_SYSTEM,
                    ),
                ],
            )

            self.previous_messages.add_messages(
                [
                    system_message,
                ]
            )

            new_messages = [
                # User
                GPTMessage(
                    role="user",
                    content=[
                        GPTMessageImageContent(  # type: ignore
                            image=image.tolist(),
                        ),
                        GPTMessageTextContent(  # type: ignore
                            text=GPT_PROMPT_SETUP.format(
                                mission=mission,
                                traj_str=traj_str,
                            ),
                        ),
                    ],
                ),
            ]
        self.previous_messages.add_messages(new_messages)

        prompt = self.previous_messages.to_prompt()
        response = self.client.chat.completions.create(
            model=VISION_MODEL,
            messages=prompt,
            max_tokens=300,
        )

        desc = response.choices[0].message.content
        self.previous_messages.add_messages(
            [
                GPTMessage(
                    role="system",
                    content=[
                        GPTMessageTextContent(  # type: ignore
                            text=desc,
                        ),
                    ],
                ),
            ],
        )

        # print(desc)

        gpt_controls = self.client.chat.completions.create(
            model=CONTROLS_MODEL,
            response_model=DroneControls,
            messages=[
                GPTMessage(
                    role="system",
                    content=[
                        GPTMessageTextContent(  # type: ignore
                            text="Generate JSON response",
                        ),
                    ],
                ).to_dict(),
                GPTMessage(
                    role="user",
                    content=[
                        GPTMessageTextContent(  # type: ignore
                            text=GPT_PROMPT_CONTROLS.format(
                                description=desc,
                                traj_str=traj_str,
                            ),
                        ),
                    ],
                ).to_dict(),
            ],
            max_tokens=300,
        )

        gpt_controls.trajectory_index += offset

        print("=" * 10)
        print(self.previous_messages.to_str(0, ["system", "user"]))
        print("=" * 10)

        print("gpt:", gpt_controls)

        return gpt_controls
