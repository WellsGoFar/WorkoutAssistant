# import cv2
# import os

output_folder = "final/test/raw_images/"
# # cap = cv2.VideoCapture("final\pushup.mp4")

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# with open("final\start_from.txt", "r") as f:
#     img_count = int(f.readline())

# while True:
#     _, frame = cap.read()
#     cv2.imwrite(output_folder + str(img_count) + ".png", frame)
#     img_count += 1
#     key = cv2.waitKey(1)

#     cv2.imshow('', frame)

#     with open("start_from.txt", "w") as f:
#         f.write(str(img_count))

#     if key == 32:
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
 
def extractFrames(pathIn, pathOut):
    # os.mkdir(pathOut)
 
    cap = cv2.VideoCapture(pathIn)
    count = 0
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame

        ret, frame = cap.read()
 
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(pathOut + "{:d}.jpg".format(count), frame)  # save frame as JPEG file
            count += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
 
def main():
    extractFrames('final\pushup.mp4', output_folder)
 
if __name__=="__main__":
    main()