# gesture_real_time_control

## Run

```bash
cd deep_control
python gesture_realtime_hand_deep.py
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
