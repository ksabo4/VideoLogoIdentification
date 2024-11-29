import cv2
import os
import numpy as np

# Function to find and draw matches
def detect_logo(test_image, logos_map):
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=5)

    for logo_name, logo_img in logos_map.items():
        
        # Convert images to grayscale
        gray_logo = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        

        # Detect SIFT keypoints and descriptors
        keypoints_logo, descriptors_logo = sift.detectAndCompute(gray_logo, None)
        keypoints_test, descriptors_test = sift.detectAndCompute(gray_test, None)

        

        

        if descriptors_logo is None or descriptors_test is None:
            print(f"Could not find match for {logo_name}")
            continue

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_logo, descriptors_test, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        matched_img = cv2.drawMatches(gray_logo, keypoints_logo, gray_test, keypoints_test,
                              good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(f"Matched Features - {logo_name}", matched_img)
        cv2.waitKey(0)
        # Ensure there are enough good matches
        if len(good_matches) > 10:
            # Extract points from good matches
            src_pts = np.float32([keypoints_logo[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography using RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)


            if M is not None:
                # Get bounding box for the logo
                h, w = gray_logo.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw the detected area on the test image
                test_image = cv2.polylines(test_image, [np.int32(dst)], True, (0, 255, 0), 3)
                print(f"Detected logo: {logo_name}")
            else:
                print(f"COULD NOT FIND LOGO")

    return test_image

# Load logos from the Logos directory
logos_dir = "Logos"
logos_map = {}
for filename in os.listdir(logos_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(logos_dir, filename)
        logos_map[filename] = cv2.imread(img_path)

# Load the test image
img_path = "TestingImages/nike3.png"
test_image = cv2.imread(img_path)


    # Detect logos in the test image
result_image = detect_logo(test_image, logos_map)

    # Show the result
cv2.imshow("Detected Logos", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
