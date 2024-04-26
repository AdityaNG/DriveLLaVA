"""
Models
"""

from typing import List

import numpy as np
from pydantic import BaseModel, validator

from drivellava.common import encode_opencv_image

from .packet import Packet


class GPTMessageContent(BaseModel):
    """GPT Message Content"""

    type: str
    data: dict


class GPTMessageTextContent(BaseModel):
    """GPT Message Text Content"""

    text: str  # "text" or "image_url"

    @property
    def type(self) -> str:
        return "text"

    @property
    def data(self) -> dict:
        return {
            "type": self.type,
            "text": self.text,
        }

    def to_content(self) -> GPTMessageContent:
        return GPTMessageContent(
            type=self.type,
            data=self.data,
        )


class GPTMessageImageContent(BaseModel):
    """GPT Message Image Content"""

    def __init__(self, image: list[list[tuple[int, int, int]]]):
        super().__init__()
        # self.__image_data = np.array(image)
        self.__image_data = image
        self.__base64_image = encode_opencv_image(np.array(self.__image_data))

    @property
    def image_url(self) -> dict:
        return {
            "url": f"data:image/jpeg;base64,{self.__base64_image}",
        }

    @property
    def type(self) -> str:
        return "image_url"

    @property
    def data(self) -> dict:
        return {
            "type": self.type,
            "image_url": self.image_url,
        }

    def to_content(self) -> GPTMessageContent:
        return GPTMessageContent(
            type=self.type,
            data=self.data,
        )


class GPTMessage(Packet):
    """GPT Message"""

    role: str  # "user" or "system"
    content: List[GPTMessageContent]

    @validator("content", each_item=True, pre=True)
    def validate_content(cls, v):
        if isinstance(v, GPTMessageTextContent) or isinstance(
            v, GPTMessageImageContent
        ):
            v = v.to_content()
        if isinstance(v, dict):
            v = GPTMessageContent(**v)
        assert isinstance(v, GPTMessageContent)
        return v

    def to_dict(self):
        return {
            "role": self.role,
            "content": [c.data for c in self.content],
        }


class GPTState(Packet):
    """GPT State"""

    messages: List[GPTMessage] = []

    def add_messages(self, messages):
        for index in range(len(messages)):
            if isinstance(
                messages[index], GPTMessageTextContent
            ) or isinstance(messages[index], GPTMessageImageContent):
                messages[index] = messages[index].to_content()
            assert isinstance(messages[index], GPTMessage)
        self.messages.extend(messages)

    def __len__(self):
        return len(self.messages)

    def pop(self, index=-1):
        return self.messages.pop(index)

    def to_prompt(self):
        return [m.to_dict() for m in self.messages]

    def to_str(self) -> str:
        """
        Returns a human readable prompt
        """
        result = ""
        for message in self.to_prompt():
            content = ""
            for cont in message["content"]:
                if cont["type"] == "image_url":
                    content += "<image>"
                elif cont["type"] == "text":
                    content += cont["text"]
                else:
                    assert False
            result += f"{message['role']}: {content}\n"

        return result
