# CSE 180 Warehouse Robot Final Project

## Team Members
- Enresto Ruiz
- Itzel Camacho
- 

## Project Description
Autonomous robot controller to detect and locate humans in a warehouse environment.

## Setup Instructions for Teammates

### 1. Clone this repository
```bash
cd ~/MRTP/MRTP/src
git clone https://github.com/YOUR_USERNAME/cse180-warehouse-project.git cse180_warehouse_project
```

### 2. Start Docker Environment
```bash
# Terminal 1: Start VNC
cd ~/MRTP/docker
make vnc

# Terminal 2: Enter container
cd ~/MRTP/docker
make bash
```

### 3. Build the workspace
```bash
cd /MRTP/MRTP
colcon build
source install/setup.bash
```

### 4. Launch simulation
```bash
ros2 launch gazeboenvs tb4_warehouse.launch.py use_rviz:=true
```

### 5. View in browser
Open: http://localhost:8080/vnc.html

### 6. Run controller (new terminal)
```bash
# New WSL terminal
cd ~/MRTP/docker
make bash

# Inside container
cd /MRTP/MRTP
source install/setup.bash
ros2 run cse180_warehouse_project warehouse_controller
```

## Git Workflow

### Making changes
```bash
git status              # See what changed
git add .               # Add all changes
git commit -m "message" # Save changes
git push                # Send to GitHub
```

### Getting teammate changes
```bash
git pull
cd /MRTP/MRTP
colcon build --packages-select cse180_warehouse_project
source install/setup.bash
```

## Robot Info
- Start: x=2.12, y=-21.3, yaw=1.57
- Topics: `/map`, `/scan`, `/amcl_pose`

## Moving Characters (for testing)
1. Click character in Gazebo
2. Open "Pose" menu
3. Change x, y
4. Press Tab
