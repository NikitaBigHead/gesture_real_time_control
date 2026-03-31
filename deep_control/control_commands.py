import math
from dataclasses import dataclass, field

from control_config import PALM_GESTURE_NAMES, ROCK_GESTURE_NAMES
from control_math import normalize_gesture_name

HAND_SLOTS = ("left", "right")


@dataclass
class CommandTrackerState:
    last_gesture_by_slot: dict
    last_fist_payload_by_slot: dict
    last_palm_payload_by_slot: dict
    palm_heading_ref_by_slot: dict


@dataclass
class CommandEvent:
    event_type: str
    hand_slot: str
    timestamp_ms: int
    gesture_from: str
    gesture_to: str
    avg_wrist_xyz: tuple[float, float, float] | None = None
    control_vector: tuple[float, float, float] | None = None
    movement_directions: list[str] = field(default_factory=list)
    yaw_deg: float | None = None


def create_command_tracker_state(hand_slots=HAND_SLOTS):
    return CommandTrackerState(
        last_gesture_by_slot={hand_slot: "unknown" for hand_slot in hand_slots},
        last_fist_payload_by_slot={hand_slot: None for hand_slot in hand_slots},
        last_palm_payload_by_slot={hand_slot: None for hand_slot in hand_slots},
        palm_heading_ref_by_slot={hand_slot: None for hand_slot in hand_slots},
    )


def reset_command_tracker_state(command_state):
    for hand_slot in command_state.last_gesture_by_slot:
        command_state.last_gesture_by_slot[hand_slot] = "unknown"
        command_state.last_fist_payload_by_slot[hand_slot] = None
        command_state.last_palm_payload_by_slot[hand_slot] = None
        command_state.palm_heading_ref_by_slot[hand_slot] = None


def _snapshot_fist_payload(hand_state):
    return {
        "avg_wrist_xyz": hand_state.avg_wrist_xyz,
        "control_vector": hand_state.control_vector,
        "movement_directions": list(hand_state.movement_directions),
    }


def _normalize_command_gesture(gesture_name):
    normalized_name = normalize_gesture_name(gesture_name)
    if normalized_name in ROCK_GESTURE_NAMES:
        return "rock"
    if normalized_name in PALM_GESTURE_NAMES:
        return "palm"
    return normalized_name


def _wrap_angle_deg(angle_deg):
    return math.degrees(math.atan2(math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))))


def _compute_palm_heading_deg(hand_landmarks):
    if hand_landmarks is None or len(hand_landmarks) <= 17:
        return None

    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]
    vx = float(pinky_mcp.x - index_mcp.x)
    vz = float(pinky_mcp.z - index_mcp.z)
    norm = math.hypot(vx, vz)
    if norm < 1e-6:
        return None

    return math.degrees(math.atan2(vx, vz))


def _append_fist_release_event(events, command_state, hand_slot, timestamp_ms, gesture_from, gesture_to):
    fist_payload = command_state.last_fist_payload_by_slot.get(hand_slot)
    if fist_payload is not None:
        events.append(
            CommandEvent(
                event_type="fist_release",
                hand_slot=hand_slot,
                timestamp_ms=timestamp_ms,
                gesture_from=gesture_from,
                gesture_to=gesture_to,
                avg_wrist_xyz=fist_payload["avg_wrist_xyz"],
                control_vector=fist_payload["control_vector"],
                movement_directions=fist_payload["movement_directions"],
            )
        )
    command_state.last_fist_payload_by_slot[hand_slot] = None


def _append_palm_release_event(events, command_state, hand_slot, timestamp_ms, gesture_from, gesture_to):
    palm_payload = command_state.last_palm_payload_by_slot.get(hand_slot)
    if palm_payload is not None:
        events.append(
            CommandEvent(
                event_type="palm_release",
                hand_slot=hand_slot,
                timestamp_ms=timestamp_ms,
                gesture_from=gesture_from,
                gesture_to=gesture_to,
                yaw_deg=palm_payload["yaw_deg"],
            )
        )
    command_state.last_palm_payload_by_slot[hand_slot] = None


def _clear_slot_state(command_state, hand_slot):
    command_state.last_gesture_by_slot[hand_slot] = "unknown"
    command_state.last_fist_payload_by_slot[hand_slot] = None
    command_state.last_palm_payload_by_slot[hand_slot] = None
    command_state.palm_heading_ref_by_slot[hand_slot] = None


def collect_command_events(frame_state, command_state, timestamp_ms):
    events = []
    active_hand_state = next((hand for hand in frame_state.hands if hand.is_active), None)
    active_hand_slot = frame_state.active_hand_slot

    for hand_slot in list(command_state.last_gesture_by_slot):
        if hand_slot != active_hand_slot:
            _clear_slot_state(command_state, hand_slot)

    if active_hand_state is None or active_hand_slot is None:
        return events

    current_gesture = _normalize_command_gesture(active_hand_state.gesture_name)
    previous_gesture = command_state.last_gesture_by_slot.get(active_hand_slot, "unknown")

    if current_gesture in ROCK_GESTURE_NAMES:
        command_state.last_fist_payload_by_slot[active_hand_slot] = _snapshot_fist_payload(active_hand_state)
    if current_gesture in PALM_GESTURE_NAMES:
        palm_heading_deg = _compute_palm_heading_deg(active_hand_state.hand_landmarks)
        palm_heading_ref_deg = command_state.palm_heading_ref_by_slot.get(active_hand_slot)
        if palm_heading_ref_deg is None and palm_heading_deg is not None:
            command_state.palm_heading_ref_by_slot[active_hand_slot] = palm_heading_deg
            palm_heading_ref_deg = palm_heading_deg

        fallback_yaw_deg = None
        if palm_heading_deg is not None and palm_heading_ref_deg is not None:
            fallback_yaw_deg = _wrap_angle_deg(palm_heading_deg - palm_heading_ref_deg)

        yaw_for_event = (
            active_hand_state.yaw_deg if active_hand_state.yaw_deg is not None else fallback_yaw_deg
        )
        if yaw_for_event is not None:
            command_state.last_palm_payload_by_slot[active_hand_slot] = {"yaw_deg": yaw_for_event}

    if previous_gesture != current_gesture:
        print("cur", current_gesture, "prev", previous_gesture)
        if previous_gesture in ROCK_GESTURE_NAMES:
            _append_fist_release_event(
                events, command_state, active_hand_slot, timestamp_ms, previous_gesture, current_gesture
            )
        if previous_gesture in PALM_GESTURE_NAMES:
            if command_state.last_palm_payload_by_slot.get(active_hand_slot) is None:
                print(
                    f"[COMMAND] palm_release skipped hand={active_hand_slot} "
                    "reason=no_yaw_payload"
                )
            _append_palm_release_event(
                events, command_state, active_hand_slot, timestamp_ms, previous_gesture, current_gesture
            )
            command_state.palm_heading_ref_by_slot[active_hand_slot] = None

    command_state.last_gesture_by_slot[active_hand_slot] = current_gesture

    return events


def format_command_event(event):
    if event.event_type == "fist_release":
        wrist_text = "n/a"
        if event.avg_wrist_xyz is not None:
            wrist_text = (
                f"{event.avg_wrist_xyz[0]:+.3f},"
                f" {event.avg_wrist_xyz[1]:+.3f},"
                f" {event.avg_wrist_xyz[2]:+.3f}"
            )

        control_text = "n/a"
        if event.control_vector is not None:
            control_text = (
                f"{event.control_vector[0]:+.3f},"
                f" {event.control_vector[1]:+.3f},"
                f" {event.control_vector[2]:+.3f}"
            )

        directions_text = "none"
        if event.movement_directions:
            directions_text = "/".join(event.movement_directions)

        return (
            f"[COMMAND] ts={event.timestamp_ms} type=fist_release hand={event.hand_slot} "
            f"gesture={event.gesture_from}->{event.gesture_to} "
            f"wrist_xyz=({wrist_text}) control_xyz=({control_text}) "
            f"directions={directions_text}"
        )

    if event.event_type == "palm_release":
        yaw_text = "n/a" if event.yaw_deg is None else f"{event.yaw_deg:+.1f} deg"
        return (
            f"[COMMAND] ts={event.timestamp_ms} type=palm_release hand={event.hand_slot} "
            f"gesture={event.gesture_from}->{event.gesture_to} yaw={yaw_text}"
        )

    return (
        f"[COMMAND] ts={event.timestamp_ms} type={event.event_type} "
        f"hand={event.hand_slot} gesture={event.gesture_from}->{event.gesture_to}"
    )


def log_command_events(events):
    for event in events:
        print(format_command_event(event))
