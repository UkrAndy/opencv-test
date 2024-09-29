class CoordinatesCalc:
    @staticmethod
    def dist_points(p1, p2):
        return  ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5

    @staticmethod
    def relative_to_real_lendmark(image, rel_points_list, pos_rect=None):
        image_width, image_height = image.shape[1], image.shape[0]
        real_landmark_points = []

        # Keypoint
        for landmark in rel_points_list:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            if pos_rect is not None:
                landmark_x += pos_rect[0][0]
                landmark_y += pos_rect[0][1]

            real_landmark_points.append([landmark_x, landmark_y])

        return real_landmark_points