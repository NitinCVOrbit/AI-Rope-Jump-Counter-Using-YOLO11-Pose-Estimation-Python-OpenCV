import cv2


def draw_jump_info(
    frame,
    jump_count,
    state
):

    # Jump count
    cv2.putText(
        frame,
        f"Jumps: {jump_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3
    )

    # Current state
    cv2.putText(
        frame,
        f"State: {state}",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    return frame