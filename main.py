import os

from Backend import SMART_EYES
import time
import os
import cv2

start_time = time.time()


sm: SMART_EYES = SMART_EYES("C:\\Users\\User\\Downloads\\f1count\\fintest2.mp4")
sm.main()
vid_path = os.listdir(sm.video_reference_path)
for rows in vid_path:
    if rows.endswith('.mp4'):
        path = os.path.realpath(rows)
        os.startfile(path)
seconds = time.time() - start_time
mins = seconds/60

print(f"--- {mins} mins ---" )


