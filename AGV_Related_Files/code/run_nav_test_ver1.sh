#!/bin/bash
#chmod +x ~/myagv_ros/src/myagv_navigation/test/run_nav_test_ver1.sh
#python3 ~/myagv_ros/src/myagv_navigation/test/nav_test_ver1.py

# SSH-compatible version: all nodes run as background processes.
# Logs are saved to ~/nav_test_logs/

source /opt/ros/noetic/setup.bash
source ~/myagv_ros/devel/setup.bash

# ── Network mode selection ───────────────────────────────────────────
# Default: AGV stand-alone (localhost)
# When connecting to Jetson: ./run_nav_test_ver1.sh jetson
if [[ "$1" == "jetson" ]]; then
    export ROS_MASTER_URI=http://192.168.1.10:11311
    export ROS_IP=192.168.1.10
    echo "Mode: AGV + Jetson (ROS_IP=192.168.1.10)"
else
    export ROS_MASTER_URI=http://localhost:11311
    unset ROS_IP
    echo "Mode: AGV alone (localhost)"
fi

LOG_DIR=~/nav_test_logs
mkdir -p "$LOG_DIR"

echo "======================================"
echo "   Starting Nav Test v1 Environment"
echo "   (SSH compatible - no gnome-terminal)"
echo "======================================"

# ===== 0. Start roscore =====
echo "[0/5] Starting roscore..."
roscore > "$LOG_DIR/roscore.log" 2>&1 &
PID_ROSCORE=$!

# Wait up to 15 seconds for roscore to become ready
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

# ===== 3. Start SLAM =====
echo "[3/5] Starting SLAM..."
roslaunch myagv_navigation myagv_slam_laser.launch > "$LOG_DIR/slam.log" 2>&1 &
PID_SLAM=$!
sleep 8

# ===== 4. Start move_base =====
echo "[4/5] Starting move_base..."
roslaunch myagv_navigation exploration_nav.launch > "$LOG_DIR/move_base.log" 2>&1 &
PID_MOVEBASE=$!
sleep 5

# ===== 5. Set parameters and start nav_test_ver1.py =====
echo "[5/5] Starting nav_test_ver1.py..."
rosparam set /nav_explore/forward_speed 0.08
rosparam set /nav_explore/turn_speed 0.45
rosparam set /nav_explore/max_angular 0.35
rosparam set /nav_explore/kp_center 0.15
rosparam set /nav_explore/kp_heading 0.10
rosparam set /nav_explore/side_error_deadband 0.08
rosparam set /nav_explore/wall_clearance 0.35
rosparam set /nav_explore/max_angular_rate 0.12
rosparam set /nav_explore/front_block_dist 0.65
rosparam set /nav_explore/front_clear_dist 0.55
rosparam set /nav_explore/side_open_dist 1.50
rosparam set /nav_explore/side_wall_min 0.20
rosparam set /nav_explore/side_wall_max 2.00
rosparam set /nav_explore/startup_grace_count 20
rosparam set /nav_explore/min_inter_turn_secs 2.0
rosparam set /nav_explore/seek_trigger_count 10
rosparam set /nav_explore/seek_check_dist 3.0
rosparam set /nav_explore/seek_timeout 35.0
rosparam set /nav_explore/min_seek_rotation 1.047
rosparam set /nav_explore/post_turn_frames 20

# ── Added parameters (bend detection / smoothing / speed coupling) ──
rosparam set /nav_explore/bend_side_thresh 0.80
rosparam set /nav_explore/bend_steer_gain 0.40
rosparam set /nav_explore/front_ema_alpha 0.35
rosparam set /nav_explore/ang_speed_reduce_thresh 0.05
rosparam set /nav_explore/wheel_k 0.12
rosparam set /nav_explore/max_wheel_speed 0.08
rosparam set /nav_explore/max_wheel_ratio 1.2
rosparam set /nav_explore/max_wheel_accel 0.002
rosparam set /nav_explore/split_detect_frames 5
rosparam set /nav_explore/split_delta_eps 0.0015
rosparam set /nav_explore/split_speed_floor 0.006
rosparam set /nav_explore/split_cooldown_secs 1.5

# ── Wall-clearance recovery parameters ─────────────────────────────
rosparam set /nav_explore/recovery_clearance 0.35
rosparam set /nav_explore/recovery_back_speed 0.05
rosparam set /nav_explore/recovery_max_secs 5.0
rosparam set /nav_explore/min_frontier_size 8
rosparam set /nav_explore/min_frontier_dist 0.3
rosparam set /nav_explore/failed_frontier_radius 0.8
rosparam set /nav_explore/nav_goal_timeout 80.0
rosparam set /nav_explore/min_explore_secs 20.0
rosparam set /nav_explore/goal_timeout 80.0
rosparam set /nav_explore/max_explore_secs 100.0
rosparam set /nav_explore/no_progress_timeout 30.0
rosparam set /nav_explore/no_progress_radius 0.5

python3 -u ~/myagv_ros/src/myagv_navigation/test/nav_test_ver1.py > "$LOG_DIR/nav_test.log" 2>&1 &
PID_NAV=$!
sleep 5

echo ""
echo "======================================"
echo "   Environment Ready"
echo "======================================"
echo ""
echo "The robot will explore autonomously:"
echo "  1. Follow corridor centreline (DT ridgeline)"
echo "  2. Stop + rotate 360 deg at obstacles to scan ahead"
echo "  3. Navigate toward unexplored frontiers only"
echo "  4. Return to start when all frontiers are exhausted"
echo "  5. Global timeout: 120 seconds"
echo "  6. No-progress watchdog: 30 seconds"
echo ""
echo "Logs: $LOG_DIR/"
echo "  tail -f $LOG_DIR/nav_test.log   (watch exploration)"
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

        # 1. Send zero velocity at 10 Hz for 1.5 s BEFORE killing the nav node.
        #    This ensures the base driver receives a stop command while ROS is
        #    still fully running.  rostopic pub -1 (one-shot) is unreliable
        #    because the base driver may not process a single message in time.
        timeout 2 rostopic pub /cmd_vel geometry_msgs/Twist \
            '{linear: {x: 0.0}, angular: {z: 0.0}}' -r 10 \
            > /dev/null 2>&1 &
        PUB_PID=$!
        sleep 1.5

        # 2. Kill the nav node now that wheels are stopped.
        kill $PID_NAV 2>/dev/null || true
        sleep 0.3
        kill $PUB_PID 2>/dev/null || true
        sleep 0.2
        echo "Saving map..."
        MAP_DIR=~/myagv_ros/src/myagv_navigation/map
        MAP_NAME="${MAP_DIR}/map"
        mkdir -p "$MAP_DIR"
        timeout 10 rosrun map_server map_saver -f "$MAP_NAME" 2>/dev/null || echo "WARNING: map_saver failed or timed out"
        sleep 2
        echo "Stopping all processes..."
        # Kill all background PIDs directly (no rosnode kill which can hang)
        kill $PID_MOVEBASE $PID_SLAM $PID_BASE $PID_LIDAR 2>/dev/null || true
        sleep 1
        kill $PID_ROSCORE 2>/dev/null || true
        # Cleanup any stragglers
        killall -9 rosmaster roscore roslaunch 2>/dev/null || true
        echo "System stopped."
        break
    fi
done