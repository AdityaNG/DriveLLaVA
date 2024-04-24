"""
GPT Vision to make Control Decisions
"""

GPT_SYSTEM = """
You are an autonomous vehicle. You are given the drone's environment state \
and you must control the vehicle to complete the mission.
"""

GPT_PROMPT_SETUP = """
Following is a visual of what the vehicle sees.

MISSION: {mission}

You can select one or more of the following actions as your next immediate \
action:
 - trajectory_index: select one of the trajectories from those drawn
 - speed_index: select a speed index from one of those provided

Trajectories: {traj_str}

Note that translation happens first, followed by rotation.
What are your next actions? Let's think step by step.
"""

GPT_PROMPT_UPDATE = """MISSION: {mission}
Updated visual is provided
What are your next actions? Let's think step by step.
"""

GPT_PROMPT_CONTROLS = """
You are assisting a pilot who is flying a drone.
Following is a description of what the driver wants to do.

Description: {description}

Trajectories: {traj_str}

Based on the description, what are the next actions the pilot should take.
You will provide the next actions in the form of a JSON object:

    "trajectory_index":     trajectory_index (str),
    "speed_index":     speed_index (str),

You can select one or more of the following actions as your next immediate \
action:
 - trajectory_index: select one of the trajectories from those drawn
 - speed_index: select a speed index from one of those provided

"""