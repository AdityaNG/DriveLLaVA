import os

from drivellava.common import (
    DEFAULT_CONTROLS_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MISSION,
    DEFAULT_VISION_MODEL,
)


def str_to_int(value: str, default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def str_to_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    return False


class DroneSettings:
    DRONE_NAME: str = os.getenv("DRONE_NAME", "airsim")


class UISettings:
    UI_ENABLED: bool = str_to_bool(os.getenv("UI_ENABLED", "True"))
    CARLA_INSTALL_PATH: str = os.getenv(
        "CARLA_INSTALL_PATH", os.path.expanduser("~/Apps/CARLA")
    )


class SystemSettings:
    SYSTEM_MISSION: str = os.getenv("SYSTEM_MISSION", DEFAULT_MISSION)
    SYSTEM_VISION_MODEL: str = os.getenv(
        "SYSTEM_VISION_MODEL", DEFAULT_VISION_MODEL
    )
    SYSTEM_CONTROLS_MODEL: str = os.getenv(
        "SYSTEM_CONTROLS_MODEL", DEFAULT_CONTROLS_MODEL
    )
    SYSTEM_LLM_PROVIDER: str = os.getenv(
        "SYSTEM_LLM_PROVIDER", DEFAULT_LLM_PROVIDER
    )
    NUM_TRAJECTORY_TEMPLATES: int = str_to_int(
        os.getenv("NUM_TRAJECTORY_TEMPLATES", "16"), 16
    )
    TRAJECTORY_SIZE: int = str_to_int(os.getenv("TRAJECTORY_SIZE", "20"), 20)
    GPT_ENABLED: bool = str_to_bool(os.getenv("GPT_ENABLED", "True"))


class Settings:
    drone: DroneSettings = DroneSettings()  # type: ignore[call-arg]
    ui: UISettings = UISettings()  # type: ignore[call-arg]
    system: SystemSettings = SystemSettings()  # type: ignore[call-arg]


settings = Settings()
