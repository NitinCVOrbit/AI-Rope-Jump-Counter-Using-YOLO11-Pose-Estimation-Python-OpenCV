import cv2  
from ultralytics import YOLO  
from jump_counter import JumpCounter  
from utils import draw_jump_info  


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():  # Define the main application function

    # ========================================================
    # LOAD VIDEO
    # ========================================================

    cap = cv2.VideoCapture("input/01.mp4")  

    if not cap.isOpened():  
        print("ERROR: Could not open video.")  
        return  

    # ========================================================
    # LOAD YOLO POSE MODEL
    # ========================================================

    model = YOLO("yolo11n-pose.pt")  

    # ========================================================
    # CREATE JUMP COUNTER
    # ========================================================

    jump_counter = JumpCounter(threshold=15, smoothing_frames=5)  

#     # ========================================================
#     # PROCESS VIDEO
#     # ========================================================

    while True:  

        ret, frame = cap.read()  

        if not ret:  
            break  

        # ----------------------------------------------------
        # RESIZE FRAME
        # ----------------------------------------------------

        frame = cv2.resize(frame, (1280, 720))  

        # ----------------------------------------------------
        # YOLO POSE DETECTION
        # ----------------------------------------------------

        results = model(frame, conf=0.5, verbose=False)  

        # ----------------------------------------------------
        # CHECK IF PERSON IS DETECTED
        # ----------------------------------------------------
        

        if len(results) > 0:
            result = results[0]

            if result.keypoints is not None:

                keypoints_xy = result.keypoints.xy.cpu().numpy()


                if len(keypoints_xy) > 0:
                    # ========================================
                    # FIRST PERSON
                    # ========================================

                    person = keypoints_xy[0]

                    # ------------------------------------------------
                    # DRAW YOLO BUILT-IN POSE VISUALIZATION
                    # ------------------------------------------------

                    frame = result.plot()  


                    # ------------------------------------------------
                    # COUNT ROPE JUMPS
                    # ------------------------------------------------

                    jump_count, state = jump_counter.update(person)  

            # ----------------------------------------------------
            # IF NO PERSON IS DETECTED
            # ----------------------------------------------------

            else:

                jump_count = jump_counter.jump_count  # Maintain the previous total jump count value
                state = jump_counter.state  # Maintain the previous jump state value

        # ----------------------------------------------------
        # DRAW JUMP INFORMATION
        # ----------------------------------------------------

        frame = draw_jump_info(frame, jump_count, state)  

        # ----------------------------------------------------
        # DISPLAY FRAME
        # ----------------------------------------------------

        cv2.imshow("Rope Jump Counter", frame)  

        # ----------------------------------------------------
        # CHECK FOR QUIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF  

        if key == ord("q"):  
            break  

    # ========================================================
    # RELEASE RESOURCES
    # ========================================================

    cap.release()  
    cv2.destroyAllWindows()  

#     # ========================================================
#     # PRINT FINAL RESULT
#     # ========================================================

#     print(f"Total Rope Jumps: {jump_counter.jump_count}")  


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__": 
    main()  