"""固定版sjtu_droneの起動処理を使い、専用worldをGUIなしで起動する。"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro
import yaml


def generate_launch_description():
    """SOS_WORLDで指定した探索worldと機体をGUIなしで起動する。"""
    upstream = get_package_share_directory("sjtu_drone_bringup")
    world = os.environ.get("SOS_WORLD", "/workspace/sos/assets/color/search.world")
    if not os.path.isfile(world):
        raise FileNotFoundError(world)
    description = get_package_share_directory("sjtu_drone_description")
    config = yaml.safe_load(Path(upstream, "config", "drone.yaml").read_text())
    config["positionXYLimit"] = -1.0
    config["velocityZLimit"] = 2.0
    config["velocityXYLimit"] = 1.2
    config_path = Path(world).parent / "mountain-controller.yaml"
    config_path.write_text(yaml.safe_dump(config))
    robot = xacro.process_file(
        os.path.join(description, "urdf", "sjtu_drone.urdf.xacro"),
        mappings={"params_path": str(config_path)},
    ).toxml()
    namespace = config["namespace"]
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("gazebo_ros"), "launch", "gzserver.launch.py")),
            launch_arguments={"world": world, "verbose": "true"}.items(),
        ),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             namespace=namespace, parameters=[{"use_sim_time": True, "robot_description": robot}]),
        Node(package="sjtu_drone_bringup", executable="spawn_drone",
             arguments=[robot, namespace], output="screen"),
    ])
