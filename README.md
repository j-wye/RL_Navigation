# RL_Navigation

How to Use:
```bash
cd && mkdir -p rl_navigation/src && cd rl_navigation
git clone https://github.com/j-wye/RL_Navigation src/
colcon build && source install/setup.bash

cd src
python3 scripts/lidar_preprocessing
python3 scripts/path_visualize.py
~/isaacsim2/python.sh scripts/train.py
```