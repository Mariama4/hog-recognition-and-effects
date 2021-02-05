try:
    import dlib
    from PIL import Image
    import cv2
    from imutils.video import VideoStream
    from imutils import face_utils, resize
    import numpy as np
except:
    import os
    os.system('pip install opencv-python')
    os.system('pip install opencv-contrib-python')
    os.system('pip install numpy')
    os.system('pip install Pillow')
    os.system('pip install dlib')
    os.system('pip install imutils')
    print('If the required library is still not installed, ',
          'then follow the instructions in the console.')

def start(source):

    vs = VideoStream(source).start()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("face_landmarks/shape_predictor_68_face_landmarks.dat")

    max_width = 700

    frame = vs.read()
    frame = resize(frame, width=max_width)

    # out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (frame.shape[1],frame.shape[0]))

    dealing = False

    while True:
        frame = vs.read()
        try:
            frame = resize(frame, width=max_width)
        except:
            exit()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(img_gray)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if dealing:
            for rect in rects:
                mask_width = rect.right() - rect.left()

                shape = predictor(img_gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[36:42]
                rightEye = shape[42:48]
                noose = shape[30:31]

                leftEyeCenter = leftEye.mean(axis=0).astype("int")

                rightEyeCenter = rightEye.mean(axis=0).astype("int")

                dY = leftEyeCenter[1] - rightEyeCenter[1]
                dX = leftEyeCenter[0] - rightEyeCenter[0]
                angle = np.rad2deg(np.arctan2(dY, dX))

                current_mask = mask.resize((mask_width*2, (int(mask_width * mask.size[1] / mask.size[0]))*2),
                                       resample=Image.LANCZOS)
                current_mask = current_mask.rotate(angle, expand=True)
                current_mask = current_mask.transpose(Image.FLIP_TOP_BOTTOM)

                noose_x = (noose[0,0] - mask_width)
                noose_y = (noose[0,1] - mask_width)

                img.paste(current_mask, (noose_x, noose_y-20), current_mask)

                frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        # out.write(frame)
        cv2.imshow("face filters", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # exit
            break
        if key == ord("q"): # filter dog
            dealing = not dealing
            if dealing == True:
                mask = Image.open("filters/filter_dog.png")
        if key == ord("w"): # filter hearts
            dealing = not dealing
            if dealing == True:
                mask = Image.open("filters/filter_hearts.png")

    # out.release()
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('Hello, using "q" and " w " to change the mask!')
    print('"esc" for exit')
    source = input('input source -> ')
    try:
        source = int(source)
        start(source)
    except:
        start(source)
    pass