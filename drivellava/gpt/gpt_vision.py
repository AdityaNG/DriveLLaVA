"""
GPT Vision to make Control Decisions
"""

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
    def __init__(self, max_history=10):
        self.client = llm_client_factory(settings.system.SYSTEM_LLM_PROVIDER)

        self.previous_messages = GPTState()
        self.max_history = max_history
        self.trajectory_encoder = TrajectoryEncoder()

    def step(
        self,
        image: np.ndarray,
        state: DroneState,
        mission: str,
    ) -> DroneControls:
        # base64_image = encode_opencv_image(image)
        traj_str = self.trajectory_encoder.get_traj_str()
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
            max_tokens=800,
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

        # If number of messages is greater than max_history,
        # remove the oldest message. Do not remove the system message
        if len(self.previous_messages) > self.max_history:
            self.previous_messages.pop(1)

        print(desc)

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

        print("gpt:", gpt_controls)

        return gpt_controls
