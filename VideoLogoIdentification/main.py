import cv2
import numpy as np

def match_features(descriptors1, descriptors2):
    good_matches = []
    ratio_thresh = 0.75

    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        sorted_indices = np.argsort(distances)

        if len(sorted_indices) > 1:
            best_match = sorted_indices[0]
            second_best_match = sorted_indices[1]

            # Apply ratio test
            if distances[best_match] < ratio_thresh * distances[second_best_match]:
                good_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_match,
                                               _distance=distances[best_match]))

    return good_matches


img1 = cv2.imread('nike1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('nike3.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

keypoint_img1 = cv2.drawKeypoints(img1, keypoints1, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint_img2 = cv2.drawKeypoints(img2, keypoints2, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Keypoints in Image 1', keypoint_img1)
cv2.imshow('Keypoints in Image 2', keypoint_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

good_matches = match_features(descriptors1, descriptors2)

matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Matched Features', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()