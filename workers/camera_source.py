import cv2 as cv
import time

def camera_image_getter(out_conn, args):
    # Camera preparation ###############################################################
    print('Camera preparation==================>>>>>>')
    print(args)
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    time.sleep(2)

    # Model load #############################################################
    while True:
        try:
            # print('=== cam Start ===')
            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break

            # Camera capture #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            # print('----===>>>> image sended')
            out_conn.send((key, ret, image))
            # print('----===>>>> image sended 2')


        except:
            print('---- Cam error ---')
            break
    print('Complete camera')