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
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  out = cv2.VideoWriter("./{}_output.mp4".format(filename), fourcc, 25.0, (1280, 720))
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret, frame = cap.read()
      if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = pose.process(frame)

      # success, image = cap.read()
      # if not success:
      #   print("Ignoring empty camera frame.")
      #   # If loading a video, use 'break' instead of 'continue'.
      #   continue

      # To improve performance, optionally mark the frame as not writeable to
      # pass by reference.
      # frame.flags.writeable = False
      # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # results = pose.process(frame)

      # Draw the pose annotation on the frame.
      frame.flags.writeable = True
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

      frame = mp_drawing.draw_landmarks(
          frame,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

      try:
        frame = cv2.resize(frame, (1280, 720))
      except Exception as e:
        print(str(e))


      out.write(frame)
      # mp_drawing.draw_landmarks(
      #     frame,
      #     results.pose_landmarks,
      #     mp_pose.POSE_CONNECTIONS,
      #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      # Flip the frame horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Pose', frame)
      # cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  out.release()
  cv2.destroyAllWindows()

pose_estimation_video("ldh.mp4")