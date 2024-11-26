import cv2
import numpy as np

img2 = cv2.imread('nike1.jpg', 0)
img1 = cv2.imread('nike3.png', 0)

orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

print("Number of keypoints in image 1:", len(keypoints1))
print("Number of keypoints in image 2:", len(keypoints2))
print("Number of matches found:", len(matches))

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print("Homography Matrix (H):")
print(H)

img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
for i in range(2):
    print(matches[i].distance)
cv2.imshow('Matched Features', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()