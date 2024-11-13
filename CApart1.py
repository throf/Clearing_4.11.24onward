import cv2
import numpy as np
import time
from scipy.spatial.distance import cdist

# Global variables to store the circle parameters averages
center_average = (0, 0)
radius_average = 0
frame_counter = 0

output_folder = r"/Users/manuelkalozi/PycharmProjects/Clearing_4.11.24onward/generated data/test_noclue"
textsheet = r"/Users/manuelkalozi/PycharmProjects/Clearing_4.11.24onward/generated data/test2.txt"
pvalue_path = r"/Users/manuelkalozi/PycharmProjects/Clearing_4.11.24onward/working_docs/Pvalue.txt"
# Use average_bgr_value_tube in the second part of your script
LOWER_GREEN2 = np.array([0, 0, 0], dtype=np.uint8)  # Default value
UPPER_GREEN2 = np.array([255, 255, 255], dtype=np.uint8)
# Use average_bgr_value_tube in the second part of your script
LOWER_YELLOW2 = np.array([0, 0, 0], dtype=np.uint8)  # Default value
UPPER_YELLOW2 = np.array([255, 255, 255], dtype=np.uint8)

# Either set the circles parameters or change the frames with the use of the set parameters
def detect_and_mark_circles(frame, frame_counter):
    print("run damc")
    global center_average, radius_average

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise and improve circle detection
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=20, param2=20, minRadius=120, maxRadius=155)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        if frame_counter < 5:
            for i, circle in enumerate(circles[0, :1]):
                center = (int(circle[0]), int(circle[1]))
                radius = circle[2]

                center_average = (
                    (frame_counter * center_average[0] + center[0]) / (frame_counter + 1),
                    (frame_counter * center_average[1] + center[1]) / (frame_counter + 1)
                )
                radius_average = (frame_counter * radius_average + radius) / (frame_counter + 1)
            print(f"circle for frame {frame_counter} calculated: Radius: {radius}, Center: {center}")

        else:
            center = center_average
            radius = radius_average - 6

            if frame_counter == 5:
                print(f"circle fully calculated: Radius: {radius}, Center: {center}")

            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), thickness=cv2.FILLED)

            # Copy the circular region from the input frame to the black frame using the mask
            frame = cv2.bitwise_and(frame, mask)

    return frame

# Either get the target_pixels from a file or set 0 so that the target_pixels will get calculated later
def get_target_pixels_from_user():
    print("run gtpfu")
    try:
        target_pixels = int(input("Enter 0 when initial step or write 'later step': "))
        print("Target will get calculated.")
        return target_pixels
    except ValueError:
        # Read the last line from the text file and store it in the variable x
        with open(pvalue_path, 'r') as file:
            lines = file.readlines()
            if lines:
                target_pixels = int(round(float(lines[-1].strip())))
                print(f"Target from last step: {target_pixels}")
            else:
                print("Error problem with file.")
        return target_pixels

# Get the yellow_pixel_count it is used to set the target_pixels
def calculate_average_yellow_area_pixel_count(frame, tube_mask):
    print("run cayapc")
    global LOWER_GREEN2  # Declare LOWER_GREEN2 as a global variable
    global LOWER_YELLOW2  # Declare LOWER_YELLOW2 as a global variable
    # Step 1: Tube area detection
    tube_area = find_tube_area(tube_mask)
    if tube_area is not None:
        average_bgr_value_tube = calculate_average_color_value(frame, tube_mask)
        print("Average BGR value in the tube area:", average_bgr_value_tube)
        # Update LOWER_GREEN2 based on the average_bgr_value_tube
        LOWER_GREEN2 = np.array([int(0.80 * value) for value in average_bgr_value_tube], dtype=np.uint8)
        # Update LOWER_GREEN2 based on the average_bgr_value_tube
        LOWER_YELLOW2 = np.array([int(0.92 * value) for value in average_bgr_value_tube], dtype=np.uint8)

    yellow_mask = calculate_yellow_mask(frame, tube_mask)
    green_mask = calculate_green_mask(frame, tube_mask)
    largest_green_area = find_green_area(green_mask)
    yellow_areas = find_yellow_areas_in_range(yellow_mask, largest_green_area, MAX_DISTANCE_TO_GREEN, MAX_DISTANCE_BETWEEN_YELLOW)

    if yellow_areas:
        pixel_counts = [cv2.contourArea(area) for area in yellow_areas]
        return sum(pixel_counts)
    else:
        return 0
# Create the mask for the analysis
def calculate_tube_mask(frame):
    print("run ctm")
    # Define the lower and upper HSV values
    tube_lower_hsv = np.array([0, 2, 10], dtype=np.uint8)    # [min_h, min_s, min_v]
    tube_upper_hsv = np.array([255, 255, 255], dtype=np.uint8)    # [max_h, max_s, max_v]

    # Create the HSV mask
    tube_mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), tube_lower_hsv, tube_upper_hsv)

    # Dilate the tube mask by one pixel
    eroded_tube_mask = cv2.erode(tube_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Create a mask for the area one pixel inside the tube outline
    one_pixel_inside_tube = eroded_tube_mask

    return one_pixel_inside_tube

# Find the tube area to limit the analysis
def find_tube_area(tube_mask):
    print("run fta")
    contours, _ = cv2.findContours(tube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None

# Calculate the average color inside the tube area
def calculate_average_color_value(frame, mask):
    print("run cacv")
    b, g, r = [frame[:, :, i][mask > 0] for i in range(3)]
    return (int(np.mean(b)), int(np.mean(g)), int(np.mean(r))) if b.size > 0 else (0, 0, 0)

# constants for getting green and yellow areas
# Constants for green color in RGB
LOWER_GREEN1 = np.array([255, 255, 255], dtype=np.uint8)
UPPER_GREEN1 = np.array([255, 255, 255], dtype=np.uint8)
# Constants for yellow color in RGB
LOWER_YELLOW1 = np.array([255, 255, 255], dtype=np.uint8)
UPPER_YELLOW1 = np.array([255, 255, 255], dtype=np.uint8)
# Constant for maximum distance from green area for marking yellow areas
MAX_DISTANCE_TO_GREEN = 8
MAX_DISTANCE_BETWEEN_YELLOW = 8

# Build a green mask by finding green areas with the use of the constants
def calculate_green_mask(frame,tube_mask):
    print("run cgm")
    mask1 = cv2.inRange(frame, LOWER_GREEN1, UPPER_GREEN1)
    mask2 = cv2.inRange(frame, LOWER_GREEN2, UPPER_GREEN2)
    green_mask = cv2.bitwise_and(cv2.bitwise_not(cv2.bitwise_or(mask1, mask2)), tube_mask)
    return green_mask

# Build a green mask by finding green areas with the use of the constants
def calculate_yellow_mask(frame,tube_mask):
    print("run cym")
    mask1 = cv2.inRange(frame, LOWER_YELLOW1, UPPER_YELLOW1)
    mask2 = cv2.inRange(frame, LOWER_YELLOW2, UPPER_YELLOW2)
    yellow_mask = cv2.bitwise_and(cv2.bitwise_not(cv2.bitwise_or(mask1, mask2)), tube_mask)
    return yellow_mask

# Get all yellow areas but return only the relevant ones
def find_yellow_areas_in_range(yellow_mask, largest_green_area, max_distance_to_green, max_distance_between_yellow):
    print("run fyair")
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or largest_green_area is None:
        return None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    accepted_yellow_areas = []
    for contour in sorted_contours:
        if is_contour_close(contour, largest_green_area, max_distance_to_green):
            # Yellow area is close enough to the green area, accept it
            accepted_yellow_areas.append(contour)
        else:
            # Check if yellow area is at least as close as MAX_DISTANCE_BETWEEN_YELLOW to another accepted yellow area
            close_enough = any(is_contour_close(contour, accepted_area, max_distance_between_yellow) for accepted_area in accepted_yellow_areas)
            if close_enough:
                accepted_yellow_areas.append(contour)
    return accepted_yellow_areas

# Calculate if distance between areas is closer than limit
def is_contour_close(contour1, contour2, max_distance):
    #print("run icc")
    if not contour1.any() or not contour2.any():
        return False
    # Find the closest points on the two contours
    closest_point_contour1, closest_point_contour2 = find_closest_points(contour1, contour2)
    # Calculate the distance between the closest points
    distance = np.sqrt((closest_point_contour2[0] - closest_point_contour1[0]) ** 2 +
                       (closest_point_contour2[1] - closest_point_contour1[1]) ** 2)
    return distance <= max_distance

# Find the two closest points of two contours
def find_closest_points(contour1, contour2):
    #print("run fcp")
    # Flatten the contours to 2D arrays
    flat_contour1 = contour1.reshape(-1, 2)
    flat_contour2 = contour2.reshape(-1, 2)
    # Compute pairwise distances between points on the two contours
    distances = cdist(flat_contour1, flat_contour2)
    # Find the indices of the minimum distance
    min_distance_indices = np.unravel_index(np.argmin(distances), distances.shape)
    # Get the closest points on the two contours
    closest_point_contour1 = flat_contour1[min_distance_indices[0]]
    closest_point_contour2 = flat_contour2[min_distance_indices[1]]
    return closest_point_contour1, closest_point_contour2

# Get all green areas but return only the biggest
def find_green_area(green_mask):
    print("run fga")
    # Find the lagest green area
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not green_contours:
        return None
    # Sort contours by area in descending order
    sorted_green_contours = sorted(green_contours, key=cv2.contourArea, reverse=True)
    # Get the largest green contour
    largest_green_area = sorted_green_contours[0]
    return largest_green_area

# Analyzing a frame with all subfunctions
def process_image_with_annotation(frame, frame_counter, target_pixels):
    print("run piwa")
    global LOWER_GREEN2  # Declare LOWER_GREEN2 as a global variable
    global LOWER_YELLOW2  # Declare LOWER_YELLOW2 as a global variable
    # Step 1: Tube area detection
    tube_mask = calculate_tube_mask(frame)
    tube_area = find_tube_area(tube_mask)
    average_bgr_value_tube = (0, 0, 0)  # Default value if tube area is not detected
    if tube_area is not None:
        average_bgr_value_tube = calculate_average_color_value(frame, tube_mask)
        print("Average BGR value in the tube area:", average_bgr_value_tube)
        # Update LOWER_GREEN2 based on the average_bgr_value_tube
        LOWER_GREEN2 = np.array([int(0.75 * value) for value in average_bgr_value_tube], dtype=np.uint8)
        # Update LOWER_GREEN2 based on the average_bgr_value_tube
        LOWER_YELLOW2 = np.array([int(0.91 * value) for value in average_bgr_value_tube], dtype=np.uint8)

    # Step 2: Yellow area analysis
    yellow_mask = calculate_yellow_mask(frame, tube_mask)
    green_mask = calculate_green_mask(frame, tube_mask)
    # Find largest green area
    largest_green_area = find_green_area(green_mask)
    # Find yellow areas within the specified distance from the largest green area
    yellow_areas = find_yellow_areas_in_range(yellow_mask, largest_green_area, MAX_DISTANCE_TO_GREEN, MAX_DISTANCE_BETWEEN_YELLOW)
    if yellow_areas:
        # Create an empty mask to store circles
        combined_mask = np.zeros_like(frame, dtype=np.uint8)
        # Set initial radius
        initial_radius = 1
        # Adjust radius to achieve black_area_pixels close to the target (700)
        radius = adjust_radius(frame, yellow_areas, initial_radius, target_pixels)
        # Draw circles using skimage.draw.disk
        for i, area in enumerate(yellow_areas):
            # Create a black mask to draw the filled area
            mask = np.zeros_like(combined_mask)

            # Draw filled area using the contour
            cv2.drawContours(mask, [area], 0, (255, 255, 255), thickness=cv2.FILLED)

            # Find the coordinates of the filled area
            filled_area_coords = np.column_stack(np.where(mask[:, :, 0] == 255))

            # Iterate over each coordinate and draw circles
            for coord in filled_area_coords:
                y, x = coord
                cv2.circle(combined_mask, (x, y), int(radius), (255, 255, 255), -1)
        # use and AND operation to get the area where combined_mask and tube_mask meet
        tube_mask_3ch = np.stack((tube_mask,) * 3, axis=-1)
        and_Combined_plus_Tube_mask = cv2.bitwise_and(combined_mask, tube_mask_3ch)
        # Find contours in the combined mask
        contours, _ = cv2.findContours(and_Combined_plus_Tube_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Combine all contours to create a single contour
            combined_contour = np.concatenate(contours)
            # Calculate the area of the region that would have been colored black
            black_area_pixels = cv2.contourArea(combined_contour)
            # only use the AND combination of combined_mask and tube_mask
            cv2.imwrite(r"/Users/manuelkalozi/PycharmProjects/Clearing_4.11.24onward/generated data/test_noclue/combimask.jpg", and_Combined_plus_Tube_mask)
            # Calculate the average RGB values of the area that would have been colored black
            average_rgb = np.mean(frame[and_Combined_plus_Tube_mask[:, :, 0] == 255], axis=0)
            # Separate the average RGB values
            average_b, average_g, average_r = average_rgb
            cv2.drawContours(frame, [combined_contour], 0, (128, 0, 128), 2)
        cv2.putText(frame, f"Pixel Count: {black_area_pixels}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"BGR: {round(average_b, 4)} {round(average_g, 4)} {round(average_r, 4)}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"Radius: {radius}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        # Sort yellow areas based on distance to green area
        yellow_areas = sorted(yellow_areas, key=lambda x: distance_to_green(x, largest_green_area))
        # Create a list to store all yellow contours
        all_yellow_areas = [area for area in yellow_areas]

        # set color of yellow area and draw them
        for i, area in enumerate(all_yellow_areas):
            # Color-coding based on the area number
            color = (0, 255 - (i - 1) * 40, 255)
            cv2.drawContours(frame, [area], -1, color, 1)
        # Draw green contour
        if largest_green_area is not None:
            cv2.drawContours(frame, [largest_green_area], -1, (0, 255, 0), 1)
            # Save average RGB values along with timestamp to a text file
            with open(textsheet, "a") as file:
                file.write(
                    f"{int(frame_counter)}\t{round(average_b, 4)}\n")
                    #f"{curr_timestamp / 1000} {round(average_b, 4)}   {round(average_g, 4)}   {round(average_r, 4)}   {black_area_pixels}\n")
        else:
            # Write timestamp with RGB averages as "-" if no yellow area is detected
            with open(textsheet, "a") as file:
                file.write(f"{int(frame_counter)}\t-\n")
                #file.write(f"{curr_timestamp / 1000}  -   -   -   -\n")
    return frame, average_bgr_value_tube  # Return the frame and the average BGR value

# Adjust radius to build black area with radius changed to get to target_pixels
def adjust_radius(frame, yellow_areas, initial_radius, target_pixels):
    print("run ar")
    best_radius = initial_radius
    closest_pixel_count = float('inf')
    checker = 0
    for factor in range(0, int(round(np.sqrt(float(target_pixels) / np.pi)))):
        radius_candidate = initial_radius + factor * 1 # *1 raise for higher speed
        print(f"{radius_candidate}")
        # Create an empty mask to store circles
        combined_mask = np.zeros_like(frame, dtype=np.uint8)
        # Draw circles using skimage.draw.disk
        for i, area in enumerate(yellow_areas):
            # Create a black mask to draw the filled area
            mask = np.zeros_like(combined_mask)

            # Draw filled area using the contour
            cv2.drawContours(mask, [area], 0, (255, 255, 255), thickness=cv2.FILLED)

            # Find the coordinates of the filled area
            filled_area_coords = np.column_stack(np.where(mask[:, :, 0] == 255))

            # Iterate over each coordinate and draw circles
            for coord in filled_area_coords:
                y, x = coord
                cv2.circle(combined_mask, (x, y), int(radius_candidate), (255, 255, 255), -1)
        # get tube_mask involved
        tube_mask = calculate_tube_mask(frame)
        tube_mask_3ch = np.stack((tube_mask,) * 3, axis=-1)
        print("combined_mask shape:", combined_mask.shape, "type:", combined_mask.dtype)
        print("tube_mask shape:", tube_mask.shape, "type:", tube_mask.dtype)
        and_Combined_plus_Tube_mask = cv2.bitwise_and(combined_mask, tube_mask_3ch)
        # Find contours in the combined mask
        contours, _ = cv2.findContours(and_Combined_plus_Tube_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Combine all contours to create a single contour
            combined_contour = np.concatenate(contours)
            # Calculate the area of the region that would have been colored black
            black_area_pixels = cv2.contourArea(combined_contour)
            # Check if the black area is within the tolerance of the target
            print(f"black area pixels: {black_area_pixels}")
            if radius_candidate == 1 and target_pixels <= black_area_pixels:
                best_radius = 1
                return best_radius

            if black_area_pixels >= (radius_average ** 2 * np.pi):
                print(f"max tube area: {radius_average ** 2 * np.pi}")
                best_radius = radius_candidate
                return best_radius

            if checker == 1 and old_distance == abs(black_area_pixels - target_pixels):
                best_radius = radius_candidate
                return best_radius

            if target_pixels >= black_area_pixels:
                old_candidate = radius_candidate
                old_distance = abs(black_area_pixels - target_pixels)
                checker = 1
            else:
                # Check if the current radius is closer to the target pixel count
                current_distance = abs(black_area_pixels - target_pixels)
                if current_distance < old_distance:
                    best_radius = radius_candidate
                    return best_radius
                else:
                    best_radius = old_candidate
                    return best_radius

# calculates the distance to green_area, used to sort yellow areas
def distance_to_green(contour, green_contour):
    print("run dtg")
    if not contour.any() or not green_contour.any():
        return float('inf')
    # Find the closest points on the two contours
    closest_point_contour, _ = find_closest_points(contour, green_contour)
    # Calculate the distance between the closest points
    distance = np.sqrt((closest_point_contour[0] - contour[:, 0, 0]) ** 2 +
                       (closest_point_contour[1] - contour[:, 0, 1]) ** 2)
    return np.min(distance)

# Define the initial frame width and height
frame_width, frame_height = 640, 480

# Define the rectangle coordinates and dimensions (1/1) is at the top left corner, y down, x right
rect_x = 1423
rect_y = 501
rect_width = 30
rect_height = 60

# Function to adjust brightness of a frame using the rectangle
def adjust_brightness(frame, target_brightness):
    print("run ab")
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the V channel (brightness component)
    v_channel = hsv_frame[:, :, 2]

    # Extract the region of interest (ROI) within the defined rectangle
    roi_v_channel = v_channel[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

    # Calculate the average V value within the ROI
    average_brightness = np.mean(roi_v_channel)

    # Calculate the brightness adjustment factor
    brightness_factor = target_brightness / average_brightness

    # Adjust the V channel of the entire frame to the target brightness
    adjusted_frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)

    # Update the V channel in the original frame with the adjusted values
    hsv_frame[:, :, 2] = adjusted_frame[:, :, 2]

    # Convert back to BGR color space
    adjusted_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

    return adjusted_frame

# Function to capture frame from webcam every second
def capture_frames():
    print("run cf")
    global frame_counter
    global target_pixels
    # Open the webcam (first available webcam)
    # Get target pixel amount from the user
    target_pixels = get_target_pixels_from_user()
    # Open the webcam (first available webcam)
    video = cv2.VideoCapture(0)  # Use index 1 for the first webcam, change as needed

    # Check if the webcam is opened successfully
    if not video.isOpened():
        print("Error: Could not open webcam")
        return

    # Initialize timestamp for frame capturing
    prev_time = time.time()

    try:
        # Wait for user input to start analysis
        input("Press Enter to start analysis...")

        frame_counter = 0  # Initialize frame counter

        while True:
            # Capture frame from the webcam
            ret, frame = video.read()

            # Check if frame is captured successfully
            if not ret:
                print("Error: Could not capture frame")
                break

            # Get current timestamp
            curr_time = time.time()

            # Check if one second has elapsed since the last frame capture
            if curr_time - prev_time >= 1:
                # Adjust the brightness of the frame
                target_brightness = 200  # out of 255 V value in HSV
                adjusted_frame = adjust_brightness(frame, target_brightness)

                # Draw rectangle on the processed frame
                cv2.rectangle(adjusted_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

                # Show the processed frame
                cv2.imshow("Processed Frame", adjusted_frame)

                # Process the first 5 frames to determine circles
                if frame_counter < 5:
                    ret, frame = video.read()
                    adjusted_frame = adjust_brightness(frame, target_brightness)
                    frame = detect_and_mark_circles(adjusted_frame, frame_counter)
                    time.sleep(1)
                    frame_counter += 1

                # After processing first 5 frames, determine target pixels if necessary
                elif target_pixels == 0:
                    yellow_areas_pixel_sum = 0
                    while 5 <= frame_counter < 10:  # Start after the 5 frames needed for setting up detect_and_mark_circles
                        ret, frame = video.read()
                        adjusted_frame = adjust_brightness(frame, target_brightness)
                        if not ret:
                            break
                        frame = detect_and_mark_circles(adjusted_frame, frame_counter)
                        curr_timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))
                        tube_mask = calculate_tube_mask(frame)
                        cv2.imwrite(r"/Users/manuelkalozi/PycharmProjects/Clearing_4.11.24onward/generated data/test_noclue/2.jpg", tube_mask)
                        print("written 2")
                        yellow_areas_pixel_singleframe = calculate_average_yellow_area_pixel_count(frame, tube_mask)
                        print(f"Yellow areas pixel count for frame {frame_counter + 1}: {yellow_areas_pixel_singleframe}")
                        cv2.imwrite(r"/Users/manuelkalozi/PycharmProjects/Clearing_4.11.24onward/generated data/test_noclue/1.jpg", frame)
                        yellow_areas_pixel_sum += yellow_areas_pixel_singleframe
                        time.sleep(1)
                        frame_counter += 1

                    # Calculate the average pixel count from the selected frames
                    average_pixel_count = yellow_areas_pixel_sum / (frame_counter - 5) if yellow_areas_pixel_sum else 0
                    target_pixels = average_pixel_count  # Use the average pixel count as target_pixels
                    with open(pvalue_path, "a") as file:
                        file.write(f"\n{int(target_pixels)}")  # Save the target_pixels to the text file

                else:
                    # Process the frame with annotations and capture the result
                    ret, frame = video.read()
                    adjusted_frame = adjust_brightness(frame, target_brightness)
                    frame = detect_and_mark_circles(adjusted_frame, frame_counter)
                    print(f"target pixels: {target_pixels}")
                    frame, average_bgr_value_tube = process_image_with_annotation(frame, frame_counter, target_pixels)
                    image_path = f"{output_folder}/frame_{frame_counter}.jpg"
                    #cv2.imwrite(image_path, frame)
                    cv2.imshow("Processed Frame", frame)
                    print(f"Saved frame at {frame_counter} ms to {image_path}")
                    time.sleep(1)
                    frame_counter += 1

                # Reset the timer after processing a frame
                prev_time = time.time()

            # Check for key press (press 'q' to exit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the webcam and close OpenCV windows
        video.release()
        cv2.destroyAllWindows()


# Main function
if __name__ == "__main__":
    # Call the function to capture frames from webcam
    capture_frames()