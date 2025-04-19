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

alias cbt='cd ~/rl_navigation && colcon build --symlink-install --parallel-workers 16 --cmake-args -DCMAKE_BUILD_TYPE=Release && sb && cd src/&& ~/isaacsim2/python.sh scripts/train.py'
alias lp='python3 ~/rl_navigation/src/scripts/lidar_preprocessing.py'
alias path='python3 ~/rl_navigation/src/scripts/path.py'
alias ts='tensorboard --logdir=~/rl_navigation/src/runs'
```