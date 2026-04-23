#!/bin/bash
#chmod +x ~/myagv_ros/src/myagv_navigation/test/run_inspect_test_ver1.sh
#python3 ~/myagv_ros/src/myagv_navigation/test/inspect_test_ver1.py

# SSH-compatible version: all nodes run as background processes.
# Logs are saved to ~/inspect_test_logs/

source /opt/ros/noetic/setup.bash
source ~/myagv_ros/devel/setup.bash

# ── Network mode selection ───────────────────────────────────────────
# Default: AGV stand-alone (localhost)
# When connecting to Jetson: ./run_inspect_test_ver1.sh jetson
if [[ "$1" == "jetson" ]]; then
    export ROS_MASTER_URI=http://192.168.1.10:11311
    export ROS_IP=192.168.1.10
    echo "Mode: AGV + Jetson (ROS_IP=192.168.1.10)"
else
    export ROS_MASTER_URI=http://localhost:11311
    unset ROS_IP
    echo "Mode: AGV alone (localhost)"
fi

LOG_DIR=~/inspect_test_logs
mkdir -p "$LOG_DIR"

echo "======================================"
echo "   Starting Inspect Patrol Environment"
echo "   (SSH compatible - no gnome-terminal)"
echo "======================================"

# ===== 0. Start roscore =====
echo "[0/5] Starting roscore..."
roscore > "$LOG_DIR/roscore.log" 2>&1 &
PID_ROSCORE=$!

for i in $(seq 1 15); do
    if rostopic list > /dev/null 2>&1; then
        echo "       roscore is running. (took ${i}s)"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "ERROR: roscore failed to start after 15s."
        echo "Log: $(cat $LOG_DIR/roscore.log 2>/dev/null || echo '(empty)')"
        kill $PID_ROSCORE 2>/dev/null
        exit 1
    fi
    sleep 1
done

MAP_DIR=~/myagv_ros/src/myagv_navigation/map
MAP_YAML="${MAP_DIR}/map.yaml"
DEFECT_YAML="${MAP_DIR}/defect_candidates.yaml"

# ── pre-flight checks ────────────────────────────────────────────────
if [ ! -f "$MAP_YAML" ]; then
    echo "ERROR: Map not found at $MAP_YAML"
    echo "Run nav_test_ver2 first to build a map."
    exit 1
fi

if [ ! -f "$DEFECT_YAML" ]; then
    echo "WARNING: No defect candidates found at $DEFECT_YAML"
    echo "The robot will navigate to start and back with no inspections."
fi

# ===== 1. Start LiDAR =====
echo "[1/5] Starting YDLidar..."
cd ~/myagv_ros/src/myagv_odometry/scripts
./start_ydlidar.sh > "$LOG_DIR/ydlidar.log" 2>&1 &
PID_LIDAR=$!
cd ~
sleep 5

# ===== 2. Start AGV base =====
echo "[2/5] Starting AGV base..."
roslaunch myagv_odometry myagv_active.launch > "$LOG_DIR/agv_base.log" 2>&1 &
PID_BASE=$!
sleep 5

# ===== 3. Start map_server + AMCL localization =====
echo "[3/5] Starting map_server + AMCL..."
roslaunch myagv_navigation navigation_active.launch map:=$MAP_YAML > "$LOG_DIR/amcl.log" 2>&1 &
PID_AMCL=$!
sleep 8

# ===== 4. Start move_base =====
echo "[4/5] Starting move_base..."
roslaunch myagv_navigation exploration_nav.launch > "$LOG_DIR/move_base.log" 2>&1 &
PID_MOVEBASE=$!
sleep 5

# ===== 5. Start inspect_test_ver1.py =====
echo "[5/5] Starting inspect_test_ver1.py..."
rosparam set /inspect_patrol/defect_yaml $DEFECT_YAML
rosparam set /inspect_patrol/goal_timeout 120.0
rosparam set /inspect_patrol/inspection_pause_secs 5.0
rosparam set /inspect_patrol/arrival_tolerance 0.3

python3 ~/myagv_ros/src/myagv_navigation/test/inspect_test_ver1.py > "$LOG_DIR/inspect_test.log" 2>&1 &
PID_INSPECT=$!
sleep 5

echo ""
echo "======================================"
echo "   Inspect Patrol Running"
echo "======================================"
echo ""
echo "The robot will:"
echo "  1. Navigate to each defect candidate location"
echo "  2. Pause for camera inspection at each point"
echo "  3. Return to start when all points visited"
echo ""
echo "Logs: $LOG_DIR/"
echo "  tail -f $LOG_DIR/inspect_test.log   (watch patrol)"
echo ""
echo "Press q to stop robot and shut everything down."
echo "======================================"

# ===== Press q to exit =====
while true
do
    read -n 1 key
    if [[ $key == "q" ]]; then
        echo ""
        echo "Stopping robot..."
        kill $PID_INSPECT 2>/dev/null || true
        sleep 0.5
        timeout 3 rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0}, angular: {z: 0.0}}' 2>/dev/null || true
        sleep 1
        echo "Stopping all processes..."
        kill $PID_MOVEBASE $PID_AMCL $PID_BASE $PID_LIDAR 2>/dev/null || true
        sleep 1
        kill $PID_ROSCORE 2>/dev/null || true
        killall -9 rosmaster roscore roslaunch 2>/dev/null || true
        echo "System stopped."
        break
    fi
done
