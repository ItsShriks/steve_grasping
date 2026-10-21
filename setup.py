import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'steve_grasping'


def tree(src):
    """data_files entries for every file under src, keeping the relative layout."""
    out = {}
    for root, _, files in os.walk(src):
        if files:
            out.setdefault(os.path.join('share', package_name, root), []).extend(
                os.path.join(root, f) for f in files)
    return list(out.items())


setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        *tree('models'),       # MuJoCo models of the tools (attached to steve_wbc's robot model)
        *tree('description'),  # Dex5-1 URDF + meshes (unitree_ros, BSD-3)
        *tree('config'),       # tool configs (config/tools/<tool>.yaml)
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ItsShriks',
    maintainer_email='nakhyeshrikar@icloud.com',
    description='Tools on the UR5e flange (camera only, Robotiq 2F-85, Dex5-1) and grasping with them',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
