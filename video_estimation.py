import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For Video input:
# For webcam input:
# cap = cv2.VideoCapture('./ldh.mp4')
# cap = cv2.VideoCapture(0)

def pose_estimation_video(filename):
  cap = cv2.VideoCapture(filename)

  if cap is None:
    print("Error opening video stream or file")

  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  out = cv2.VideoWriter("./{}_output.mp4".format(filename), fourcc, 25.0, (1280, 720))
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if success == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      image = mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

      try:
        image = cv2.resize(image, (1280, 720))
      except Exception as e:
        print(str(e))

      wout = out.write(image)

      # Flip the image horizontally for a selfie-view display.
      if success and image is not None:
        if image.shape[0] > 0 and image.shape[1] > 0:
          cv2.imshow('MediaPipe Pose', image)
      else:
        print("Ignoring empty image.")
      # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  wout.release()
  cv2.destroyAllWindows()

pose_estimation_video("OMG.mp4")