HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

P0_MOVING_AVG_WINDOW = 10
XY_DEAD_ZONE = 0.03
Z_DEAD_ZONE = 0.02
XY_MAX_DELTA = 0.30
Z_MAX_DELTA = 0.12
TURN_DEAD_ZONE_DEG = 10

PALM_GESTURE_NAMES = {"palm", "open_palm"}
ROCK_GESTURE_NAMES = {"rock", "closed_fist"}
NONE_GESTURE_NAMES = {"none", "unknown"}
