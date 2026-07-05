"""Plain-Python stand-ins for the ROS2 geometry_msgs.msg types RDP type-hints
against. See pyproject.toml in this stub package for why these exist.
"""


class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class Quaternion:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class Pose:
    def __init__(self, position: Point = None, orientation: Quaternion = None):
        self.position = position if position is not None else Point()
        self.orientation = orientation if orientation is not None else Quaternion()
