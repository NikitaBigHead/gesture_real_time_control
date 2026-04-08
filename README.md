# gesture_real_time_control

## Run

```bash
cd deep_control
python gesture_realtime_hand_deep.py
```

## RAG Interaction Server

`rag_interaction.py` can now run in three modes:

- `--mode client` sends gesture commands to `/command` and spoken text to `/query`.
- `--mode server` starts a minimal FastAPI server with those two endpoints.
- `--mode video-client` streams RGB video over ZMQ to an RGB receiver.

Run the server:

```bash
python rag_interaction.py --mode server --host 0.0.0.0 --port 8000
```

Run the video client:

```bash
python rag_interaction.py --mode video-client --video-server-host 127.0.0.1 --video-preview
```

To make `--mode client` receive gesture frames from that streamed video instead of a local RealSense camera:

```bash
python rag_interaction.py --mic-id 3 --gesture-video-source stream --video-bind-host 0.0.0.0 --video-port 5550
```

Endpoints:

- `POST /command` - accepts `{"command": "left" | "right" | "close" | "open"}`
- `POST /query` - accepts `{"query": "..." }`
- `GET /health` - health check.

If FastAPI is not installed yet:

```bash
/home/dzmitry/gesture_real_time_control/.venv/bin/pip install fastapi uvicorn
```

## deep_control

`deep_control` is the main real-time gesture control module. It gets RGB and depth data from RealSense, detects hands and body pose with MediaPipe, computes the hand control state, and sends commands to the drone through ROS 2 / MAVROS when needed.

Data flow:

1. `gesture_realtime_hand_deep.py` starts `control_runtime.main()`.
2. `control_runtime.py` starts the MediaPipe Gesture Recognizer, Pose Landmarker, RealSense pipeline, and visualization.
3. `control_state.py` computes the smoothed gesture, averaged wrist position, `control_vector`, movement directions, and yaw for the active hand.
4. `control_commands.py` converts gesture changes into `fist_release` and `palm_release` events.
5. `gesture_drone_node.py` receives these events and publishes velocity commands to MAVROS.

Directory contents:

- `gesture_realtime_hand_deep.py` - entry point for the pipeline.
- `control_runtime.py` - main runtime: CLI arguments, model startup, frame reading, processing loop, and ROS integration.
- `control_state.py` - computes the frame state and active hand state, smooths gestures and yaw, and selects the hand used for control.
- `control_commands.py` - creates high-level control events from the gesture sequence.
- `gesture_drone_node.py` - ROS 2 node that sends `TwistStamped` messages and handles translation and yaw commands.
- `control_geometry.py` - matches the hand to the left or right arm using pose landmarks and computes forearm and palm vectors.
- `control_depth.py` - reads and median-filters depth around the wrist pixel.
- `control_math.py` - vector math, dead zones, EMA, and `control_vector` computation.
- `control_overlay.py` - visual overlay: hand skeleton, tracking status, movement directions, and yaw.
- `control_config.py` - thresholds, smoothing windows, dead zones, and gesture name sets.
- `gesture_recognizer.task` - MediaPipe gesture recognition model.
- `pose_landmarker_heavy.task` - main MediaPipe pose tracking model.
- `pose_landmarker_full.task` - alternative pose model.

Current control logic:

- `rock` / `closed_fist` are used for linear motion relative to the hand start point.
- `palm` / `open_palm` are used to estimate yaw from the palm direction relative to the forearm.
- A command is not sent on every frame. It is sent as an event when the user releases the related gesture.

## Yaw Troubleshooting

If translation works but yaw does not, check the full chain:

1. `palm_release` event must be generated.
2. `yaw_deg` must be present in that event.
3. Drone pose topic must provide orientation (for current yaw estimate).
4. MAVROS must accept `TwistStamped.twist.angular.z` on `setpoint_velocity/cmd_vel`.

Quick checks:

```bash
# Run detector and look for palm_release logs in terminal
cd deep_control
python gesture_realtime_hand_deep.py

# In another terminal: verify pose topic publishes orientation
ros2 topic echo /drone2/mavros/local_position/pose --once

# Verify cmd_vel includes angular.z during a palm_release rotation
ros2 topic echo /drone2/mavros/setpoint_velocity/cmd_vel
```

Notes:

- Default pose topic is now `/drone2/mavros/local_position/pose`.
- Rotation is event-based: yaw is queued when gesture transitions from `palm/open_palm` to another gesture.
- If your setup uses different namespace, pass it explicitly:

```bash
python gesture_realtime_hand_deep.py \
  --mavros-prefix /<your_ns>/mavros \
  --drone-pose-topic /<your_ns>/mavros/local_position/pose
```







 python3 gesture_realtime_hand_deep.py   --mavros-prefix /mavros   --drone-pose-topic /mavros/local_position/pose
