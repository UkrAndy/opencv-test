import mediapipe as mp

def pose_getter(in_conn, out_conn, args):
    # Model load #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=args.use_static_image_mode,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    # ----------------

    while True:
        image = in_conn[1].recv()
        # print('==== process pose start ======')
        results_pose = pose.process(image)
        if results_pose.pose_landmarks is not None and len(results_pose.pose_landmarks.landmark) > 0:
            # print('ok ok ok ok ok ok ok ok ok ok ok ok ')
            # print(results_pose.pose_landmarks.landmark)
            out_conn[0].send(list(results_pose.pose_landmarks.landmark))
        else:
            out_conn[0].send([])
        # print('==== process pose end ======')
