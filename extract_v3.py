import numpy as np
import cv2
import math

# Constants for filtering and thresholding
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
MIN_HANDWRITING_HEIGHT_PIXEL = 20

# Features defined as global variables
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0

def resize_and_crop(image, size=(800, 800)):
    h, w = image.shape[:2]
    target_w, target_h = size


    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)


    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    cropped_image = resized_image[top:top + target_h, left:left + target_w]

    return cropped_image
def apply_bilateral_filter(image, diameter):
    return cv2.bilateralFilter(image, diameter, 50, 50)


def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)


def apply_threshold(image, threshold_value):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return binary_image


def apply_dilation(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def apply_erosion(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def straighten_image(image):
    global BASELINE_ANGLE

    total_angle = 0.0
    contour_count = 0

    filtered_image = apply_bilateral_filter(image, 3)
    thresholded_image = apply_threshold(filtered_image, 120)
    dilated_image = apply_dilation(thresholded_image, (5, 100))

    contours, _ = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if height > width or height < MIN_HANDWRITING_HEIGHT_PIXEL:
            continue

        region_of_interest = image[y:y + height, x:x + width]

        if width < image.shape[1] / 2:
            region_of_interest.fill(255)
            image[y:y + height, x:x + width] = region_of_interest
            continue

        rect = cv2.minAreaRect(contour)
        center = rect[0]
        angle = rect[2]

        if angle < -45.0:
            angle += 90.0

        rotation_matrix = cv2.getRotationMatrix2D(((x + width) / 2, (y + height) / 2), angle, 1)
        straightened_region = cv2.warpAffine(region_of_interest, rotation_matrix, (width, height),
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        image[y:y + height, x:x + width] = straightened_region

        total_angle += angle
        contour_count += 1

    if contour_count == 0:
        mean_angle = total_angle
    else:
        mean_angle = total_angle / contour_count

    BASELINE_ANGLE = mean_angle
    return image


def calculate_horizontal_projection(image):
    height, width = image.shape[:2]
    row_sums = [np.sum(image[row:row + 1, 0:width]) for row in range(height)]
    return row_sums


def calculate_vertical_projection(image):
    height, width = image.shape[:2]
    column_sums = [np.sum(image[0:height, col:col + 1]) for col in range(width)]
    return column_sums

def extract_lines(image):
    global LETTER_SIZE
    global LINE_SPACING
    global TOP_MARGIN

    # Apply bilateral filter
    filtered_image = apply_bilateral_filter(image, 5)

    # Convert to grayscale and binarize the image using INVERTED binary thresholding
    thresholded_image = apply_threshold(filtered_image, 160)

    # Extract the horizontal projection of the image
    horizontal_projection_list = calculate_horizontal_projection(thresholded_image)

    # Extracting 'Top Margin' feature
    top_margin_count = 0
    for row_sum in horizontal_projection_list:
        if row_sum <= 255:
            top_margin_count += 1
        else:
            break

    # Initialize variables for extracting lines and spaces
    line_top = 0
    line_bottom = 0
    space_top = 0
    space_bottom = 0
    index_count = 0
    set_line_top = True
    set_space_top = True
    include_next_space = True
    space_gaps = []
    lines = []

    # Scan through the horizontal projection
    for i, row_sum in enumerate(horizontal_projection_list):
        if row_sum == 0:
            if set_space_top:
                space_top = index_count
                set_space_top = False
            index_count += 1
            space_bottom = index_count
            if i < len(horizontal_projection_list) - 1:
                if horizontal_projection_list[i + 1] == 0:
                    continue
            if include_next_space:
                space_gaps.append(space_bottom - space_top)
            else:
                if not space_gaps:
                    previous_space = 0
                else:
                    previous_space = space_gaps.pop()
                space_gaps.append(previous_space + space_bottom - line_top)
            set_space_top = True
        elif row_sum > 0:
            if set_line_top:
                line_top = index_count
                set_line_top = False
            index_count += 1
            line_bottom = index_count
            if i < len(horizontal_projection_list) - 1:
                if horizontal_projection_list[i + 1] > 0:
                    continue
                if line_bottom - line_top < 20:
                    include_next_space = False
                    set_line_top = True
                    continue
            include_next_space = True
            lines.append([line_top, line_bottom])
            set_line_top = True

    fine_lines = []
    for line in lines:
        anchor = line[0]
        anchor_points = []
        up_hill = True
        down_hill = False
        segment = horizontal_projection_list[line[0]:line[1]]

        for row_sum in segment:
            if up_hill:
                if row_sum < ANCHOR_POINT:
                    anchor += 1
                    continue
                anchor_points.append(anchor)
                up_hill = False
                down_hill = True
            if down_hill:
                if row_sum > ANCHOR_POINT:
                    anchor += 1
                    continue
                anchor_points.append(anchor)
                down_hill = False
                up_hill = True

        if len(anchor_points) < 2:
            continue

        line_top = line[0]
        for x in range(1, len(anchor_points) - 1, 2):
            line_mid = (anchor_points[x] + anchor_points[x + 1]) // 2
            line_bottom = line_mid
            if line_bottom - line_top < 20:
                continue
            fine_lines.append([line_top, line_bottom])
            line_top = line_bottom
        if line[1] - line_top < 20:
            continue
        fine_lines.append([line_top, line[1]])

    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_with_midzone_count = 0
    has_midzone = False
    for line in fine_lines:
        segment = horizontal_projection_list[line[0]:line[1]]
        for row_sum in segment:
            if row_sum < MIDZONE_THRESHOLD:
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                has_midzone = True
        if has_midzone:
            lines_with_midzone_count += 1
            has_midzone = False

    if lines_with_midzone_count == 0:
        lines_with_midzone_count = 1

    total_space_row_count = space_nonzero_row_count + np.sum(space_gaps[1:-1])
    average_line_spacing = float(total_space_row_count) / lines_with_midzone_count
    average_letter_size = float(midzone_row_count) / lines_with_midzone_count
    LETTER_SIZE = average_letter_size

    if average_letter_size == 0:
        average_letter_size = 1
    relative_line_spacing = average_line_spacing / average_letter_size
    LINE_SPACING = relative_line_spacing
    relative_top_margin = float(top_margin_count) / average_letter_size
    TOP_MARGIN = relative_top_margin

    return fine_lines

def extract_words(image, lines):
    global LETTER_SIZE
    global WORD_SPACING

    # Apply bilateral filter
    filtered_image = apply_bilateral_filter(image, 5)

    # Convert to grayscale and binarize the image using INVERTED binary thresholding
    thresholded_image = apply_threshold(filtered_image, 180)

    # Width of the document
    document_width = thresholded_image.shape[1]
    space_gaps = []  # Stores the amount of space between words
    words = []  # A 2D list storing the coordinates of each word: y1, y2, x1, x2

    # Isolated words/components are extracted from each line by analyzing the vertical projection
    for line in lines:
        extract = thresholded_image[line[0]:line[1], 0:document_width]  # y1:y2, x1:x2
        vertical_projection_list = calculate_vertical_projection(extract)

        word_start = 0
        word_end = 0
        space_start = 0
        space_end = 0
        index_count = 0
        set_word_start = True
        set_space_start = True
        spaces = []

        # Scan the vertical projection
        for j, col_sum in enumerate(vertical_projection_list):
            if col_sum == 0:
                if set_space_start:
                    space_start = index_count
                    set_space_start = False
                index_count += 1
                space_end = index_count
                if j < len(vertical_projection_list) - 1:
                    if vertical_projection_list[j + 1] == 0:
                        continue
                if (space_end - space_start) > int(LETTER_SIZE / 2):
                    spaces.append(space_end - space_start)
                set_space_start = True
            elif col_sum > 0:
                if set_word_start:
                    word_start = index_count
                    set_word_start = False
                index_count += 1
                word_end = index_count
                if j < len(vertical_projection_list) - 1:
                    if vertical_projection_list[j + 1] > 0:
                        continue
                row_count = 0
                for k in range(line[1] - line[0]):
                    row_segment = thresholded_image[line[0] + k:line[0] + k + 1, word_start:word_end]  # y1:y2, x1:x2
                    if np.sum(row_segment):
                        row_count += 1
                if row_count > int(LETTER_SIZE / 2):
                    words.append([line[0], line[1], word_start, word_end])
                set_word_start = True

        space_gaps.extend(spaces[1:-1])

    total_space_columns = np.sum(space_gaps)
    space_count = len(space_gaps)
    if space_count == 0:
        space_count = 1
    average_word_spacing = float(total_space_columns) / space_count
    if LETTER_SIZE == 0:
        relative_word_spacing = average_word_spacing
    else:
        relative_word_spacing = average_word_spacing / LETTER_SIZE
    WORD_SPACING = relative_word_spacing

    return words

def extract_slant(img, words):
    global SLANT_ANGLE

    # Predefined angles in radians for slant detection
    theta = [-0.785398, -0.523599, -0.261799, -0.0872665, 0.01, 0.0872665, 0.261799, 0.523599, 0.785398]

    # Initialize s_function and count_ lists for each angle
    s_function = [0.0] * len(theta)
    count_ = [0] * len(theta)

    # Apply bilateral filter
    filtered_image = apply_bilateral_filter(img, 5)

    # Convert to grayscale and binarize the image using INVERTED binary thresholding
    thresholded_image = apply_threshold(filtered_image, 180)

    # Loop for each value of angle in theta
    for i, angle in enumerate(theta):
        s_temp = 0.0  # Overall sum of the functions of all the columns of all the words
        count = 0  # Count of columns considered to contain a vertical stroke

        # Loop for each word
        for word in words:
            original_word = thresholded_image[word[0]:word[1], word[2]:word[3]]  # y1:y2, x1:x2

            height = word[1] - word[0]
            width = word[3] - word[2]

            # Calculate the shift for affine transformation
            shift = (math.tan(angle) * height) / 2

            # Calculate the amount of extra space needed to preserve information
            pad_length = abs(int(shift))

            # Create a new image that can perfectly hold the transformed image
            blank_image = np.zeros((height, width + pad_length * 2), np.uint8)
            new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            new_image[:, pad_length:width + pad_length] = original_word

            # Points for affine transformation
            (height, width) = new_image.shape[:2]
            pts1 = np.float32([[width / 2, 0], [width / 4, height], [3 * width / 4, height]])
            pts2 = np.float32([[width / 2 + shift, 0], [width / 4 - shift, height], [3 * width / 4 - shift, height]])
            M = cv2.getAffineTransform(pts1, pts2)
            deslanted_image = cv2.warpAffine(new_image, M, (width, height))

            # Find the vertical projection on the transformed image
            vertical_projection_list = calculate_vertical_projection(deslanted_image)

            # Loop for each column in the word image
            for k, col_sum in enumerate(vertical_projection_list):
                if col_sum == 0:
                    continue

                num_fg_pixels = col_sum / 255

                if num_fg_pixels < int(height / 3):
                    continue

                column = deslanted_image[0:height, k:k + 1].flatten()

                for l, pixel in enumerate(column):
                    if pixel == 0:
                        continue
                    break

                for m, pixel in enumerate(column[::-1]):
                    if pixel == 0:
                        continue
                    break

                delta_y = height - (l + m)
                h_sq = (float(num_fg_pixels) / delta_y) ** 2
                h_weighted = (h_sq * num_fg_pixels) / height

                s_temp += h_weighted
                count += 1

        s_function[i] = s_temp
        count_[i] = count

    max_value = max(s_function)
    max_index = s_function.index(max_value)

    if max_index == 0:
        angle = 45
        result = " : Extremely right slanted"
    elif max_index == 1:
        angle = 30
        result = " : Above average right slanted"
    elif max_index == 2:
        angle = 15
        result = " : Average right slanted"
    elif max_index == 3:
        angle = 5
        result = " : A little right slanted"
    elif max_index == 5:
        angle = -5
        result = " : A little left slanted"
    elif max_index == 6:
        angle = -15
        result = " : Average left slanted"
    elif max_index == 7:
        angle = -30
        result = " : Above average left slanted"
    elif max_index == 8:
        angle = -45
        result = " : Extremely left slanted"
    elif max_index == 4:
        p = s_function[4] / s_function[3] if s_function[3] != 0 else s_function[4]
        q = s_function[4] / s_function[5] if s_function[5] != 0 else s_function[4]

        if ((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)) or ((p <= 1.2 and q - p > 0.4) or (q <= 1.2 and p - q > 0.4)):
            angle = 0
            result = " : No slant"
        else:
            max_index = 9
            angle = 180
            result = " : Irregular slant behaviour"

        if angle == 0:
            print("\n************************************************")
            print("Slant determined to be straight.")
        else:
            print("\n************************************************")
            print("Slant determined to be irregular.")

    SLANT_ANGLE = angle
    return

def barometer(image):
    global PEN_PRESSURE

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the image pixel by pixel
    inverted_image = 255 - gray_image

    # Apply bilateral filtering
    filtered_image = apply_bilateral_filter(inverted_image, 3)

    # Binary thresholding to zero
    _, thresh_image = cv2.threshold(filtered_image, 100, 255, cv2.THRESH_TOZERO)

    # Calculate total intensity and count of non-zero pixels
    non_zero_pixels = thresh_image[thresh_image > 0]
    total_intensity = non_zero_pixels.sum()
    pixel_count = len(non_zero_pixels)

    # Calculate average intensity
    average_intensity = float(total_intensity) / pixel_count if pixel_count != 0 else 0
    PEN_PRESSURE = average_intensity

    return

def start(file_name):

    global BASELINE_ANGLE
    global TOP_MARGIN
    global LETTER_SIZE
    global LINE_SPACING
    global WORD_SPACING
    global PEN_PRESSURE
    global SLANT_ANGLE

    # Read the image from disk
    image = cv2.imread(f'uploads/{file_name}')
    if image is not None:
        image = resize_and_crop(image, size=(800, 800))
        cv2.imwrite(f"/cropped/{file_name}", image)
    else:
        print("Image not found or unable to load.")

    # Extract pen pressure
    barometer(image)

    # Apply contour operation to straighten the image
    straightened = straighten_image(image)

    # Extract lines of handwritten text from the straightened image
    line_indices = extract_lines(straightened)

    # Extract words from each line using vertical projection
    word_coordinates = extract_words(straightened, line_indices)

    # Extract average slant angle of all the words containing a long vertical stroke
    extract_slant(straightened, word_coordinates)

    # Round the extracted features to 2 decimal places
    BASELINE_ANGLE = round(BASELINE_ANGLE, 2)
    TOP_MARGIN = round(TOP_MARGIN, 2)
    LETTER_SIZE = round(LETTER_SIZE, 2)
    LINE_SPACING = round(LINE_SPACING, 2)
    WORD_SPACING = round(WORD_SPACING, 2)
    PEN_PRESSURE = round(PEN_PRESSURE, 2)
    SLANT_ANGLE = round(SLANT_ANGLE, 2)

    # Return the extracted features
    return [
        BASELINE_ANGLE,
        TOP_MARGIN,
        LETTER_SIZE,
        LINE_SPACING,
        WORD_SPACING,
        PEN_PRESSURE,
        SLANT_ANGLE
    ]
