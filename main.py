import cv2
import argparse
import numpy as np
import math, sys
from ocr import to_text
from moviepy.editor import VideoFileClip


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=False, help="Path to image")
ap.add_argument("-m", "--multiple", type=str, required=False, help="Path to folder for multiple images")
ap.add_argument("-v", "--video", type=str, required=False, help="Path to video")
args = vars(ap.parse_args())


class Character:
    def __init__(self, cnt, i, shape):
        self.id = i
        self.cnt = cnt
        [x, y, w, h] = cv2.boundingRect(cnt)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.boundRect = [x, y, w, h]
        self.area = w*h

        self.centerX = (x+x+w)/2
        self.centerY = (y+y+h)/2

        self.diagonalSize = math.sqrt((w**2) + (h**2))

        self.aspectRatio = float(w)/float(h)

        self.img_width = float(shape[1])
        self.img_height = float(shape[0])
        self.img_area = float(shape[1]*shape[0])

        self.char = is_char(self)


# def is_char(char):
#     if (0.00005*char.img_area) < char.area < (0.0007*char.img_area) \
#             and (0.0005*char.img_width) < char.w < (0.06*char.img_width) \
#             and (0.012*char.img_height) < char.h < (0.03*char.img_height) \
#             and 0.2 < char.aspectRatio < 0.9:
#         return True
#     else:
#         return False
def is_char(char):
    if (0.00005*char.img_area) < char.area < (0.0014*char.img_area) \
            and (0.0005*char.img_width) < char.w < (0.11*char.img_width) \
            and (0.012*char.img_height) < char.h < (0.09*char.img_height) \
            and 0.2 < char.aspectRatio < 0.9:
        return True
    else:
        return False


# Helpers
def resize(image_in, scale):
    width = int(image_in.shape[1] * scale)
    height = int(image_in.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image_in, dim)


def getKey(item):
    return item[0]


def sort_coords(l):
    return sorted(l, key=getKey)


def list_pick(indexes, list_in):
    out = []
    for i in indexes:
        out.append(list_in[i])
    return out


def get_relative_coords(points):
    min_x = 9999
    min_y = 9999

    for c in points:
        if c[0] < min_x:
            min_x = c[0]
        if c[1] < min_y:
            min_y = c[1]

    p = points
    out = [[p[0][0]-min_x, p[0][1]-min_y], [p[1][0]-min_x, p[1][1]-min_y], [p[2][0]-min_x,
                                                                            p[2][1]-min_y],
           [p[3][0]-min_x, p[3][1]-min_y]]
    return out


# Transformers
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = (255, 255, 255)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def process_image(image_in):
    # Split up image to get raw values
    if args['video']:
        hsv = cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV)
    else:
        hsv = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)

    _, _, value = cv2.split(hsv)

    # Kernel to morph picture
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Use tophat/blackhat operations
    top_hat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

    # Add and subtract between morph operations
    add = cv2.add(value, top_hat)
    subtract = cv2.subtract(add, black_hat)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(subtract, (5, 5), 0)

    # Apply threshold --------- Tweak this
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 3)

    return thresh


def contours(image_in):
    # cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
    cv2MajorVersion = cv2.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) >= 4:
        contours, hierarchy = cv2.findContours(image_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        imageContours, contours, hierarchy = cv2.findContours(image_in, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def characters(contours, shape):
    characters_full = []

    for i in range(len(contours)):
        char = Character(contours[i], i, shape)
        if char.char:
            characters_full.append(char)

    return characters_full


def adjeacent(characters):
    out = []
    related = []
    for i1 in range(len(characters)):
        c1 = characters[i1]
        for i2 in range(len(characters)):
            c2 = characters[i2]

            # area_ratio = abs(float(c1.area)/float(c2.area))
            # width_ratio = abs(float(c1.w)/float(c2.w))
            height_ratio = abs(float(c1.h)/float(c2.h))
            x_delta = abs(c1.x-c2.x)
            y_delta = abs(c1.y-c2.y)

            if (c1.img_width*0.01) < x_delta < (c1.img_width*0.05) \
                and (c1.img_height * 0.0) < y_delta < (c1.img_height * 0.01) \
                    and 0.6 < height_ratio < 1.3:
                related.append((i1, i2))

    # WIP
    groups = []
    for tup in related:
        grouped = False
        if len(groups) > 0:
            for i in range(len(groups)):
                group = groups[i]
                if tup[0] in group and tup[1] not in group:
                    groups[i].append(tup[1])
                    grouped = True
                elif tup[1] in group and tup[0] not in group:
                    groups[i].append(tup[0])
                    grouped = True
                elif tup[1] in group and tup[0] in group:
                    grouped = True
            if not grouped:
                groups.append([tup[0], tup[1]])

        else:
            groups.append([tup[0], tup[1]])

    # return list_pick(groups[11], characters)

    # flat_list = []
    for group in groups:
        max_x = 0
        min_x = 9999
        max_y = 0
        min_y = 9999

        if 6 <= len(group) <= 8:
            # print(len(group))
            for item in group:
                char = characters[item]

                if (char.x+char.w) > max_x:
                    max_x = (char.x+char.w)
                if char.x < min_x:
                    min_x = char.x
                if (char.y+char.h) > max_y:
                    max_y = (char.y+char.h)
                if char.y < min_y:
                    min_y = char.y

            ratio = abs(float(max_x-min_x)/float(max_y-min_y))
            # print(ratio)
            # if 3.4 < ratio < 5.4:
            if 2.5 < ratio < 7.0:
                out = list_pick(group, characters)
                return out
    if not out:
        return out


def object_to_cnts(contours):
    out = []
    for contour in contours:
        out.append(contour.cnt)
    return out


# Plate editing
def straighten_plate(plate, img):
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, bl, br, tr) = plate
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1],
        [maxWidth - 1, 0]], np.float32)

    rect = np.array(get_relative_coords(plate), np.float32)
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp


def bounding_shape(chars):
    points = []
    for c in chars:
        points.append([c.x, c.y, (c.x+c.w), (c.y+c.h)])

    sc = sort_coords(points)
    shape = [[sc[0][0], sc[0][1]], [sc[0][0], sc[0][3]], [sc[-1][2], sc[-1][3]], [sc[-1][2], sc[-1][1]]]
    return shape


def crop_plate(input_img, shape, padding):
    shape = add_padding(shape, padding)
    pts = np.array([shape], np.int32)
    img = region_of_interest(input_img, pts)
    box = bounding_box(shape)
    return img[box[0][1]: box[1][1], box[0][0]: box[1][0]], pts, shape


def add_padding(s, percent):
    width = ((s[2][0]+s[3][0])/2)-((s[0][0]+s[1][0])/2)
    p = int(width*percent)
    out = [[], [], [], []]
    out[0] = [s[0][0]-p, s[0][1]-p]
    out[1] = [s[1][0]-p, s[1][1]+p]
    out[2] = [s[2][0]+p, s[2][1]+p]
    out[3] = [s[3][0]+p, s[3][1]-p]
    return out


# Input: characters and padding
def bounding_box(points):
    minX = 9999
    minY = 9999
    maxX = 0
    maxY = 0

    for c in points:
        if c[0] < minX and c[1] < minY:
            minX = c[0]
            minY = c[1]
        if c[1] > maxY and c[0] > maxX:
            maxX = c[0]
            maxY = c[1]

    box = [(minX, minY), (maxX, maxY)]
    return box


# Main action starts here!
# setup



def toggle_images(key):
    global time_clip, show, time_frame, duration
    # print(time_clip)
    if key == 83:
        time_clip += time_frame
    elif key == 81:
        time_clip -= time_frame

    if (time_clip >= (duration-time_frame)):
        time_clip = 0
        print("Reached end of file!")
    if key == 116:
        print(time_clip)
    if key == 113:
        show = False
        return

    sys.stdout.flush()


def show_processed(image_raw, i):
    # Process
    resized = resize(image_raw, image_in_scale)

    height = resized.shape[0]
    width = resized.shape[1]
    if args['video']:
        region_of_interest_vertices = [
            (width*0, height*0.2),
            (width*1, height*0.2),
            (width*1, height*1),
            (width*0, height*1)
        ]
    else:
        region_of_interest_vertices = [
            (width*0, height*0.5),
            (width*1, height*0.5),
            (width*1, height*1),
            (width*0, height*1)
        ]

    thresh = process_image(resized)


    regionalized_image = region_of_interest(thresh, np.array(
        [region_of_interest_vertices], np.int32),)

    raw_contours = contours(regionalized_image)
    height, width = regionalized_image.shape
    contours_image = np.zeros((height, width, 3), dtype=np.uint8)
    test_image = np.zeros((height, width, 3), dtype=np.uint8)

    # cv2.drawContours(contours_image, raw_contours, -1, (255, 255, 255))
    # cv2.imshow("Contours", resize(contours_image, im_show_scale))

    characters_full = characters(raw_contours, regionalized_image.shape)
    # print(len(characters_full))
    text = ""
    if len(characters_full) > 0:
        adjeacent_characters = adjeacent(characters_full)
        if len(adjeacent_characters) > 0:
            adjeacent_characters_isolated = object_to_cnts(adjeacent_characters)
            cv2.drawContours(contours_image, adjeacent_characters_isolated, -1, (255, 255, 255))
            cv2.imshow("Contours", resize(contours_image, im_show_scale))

            shape = bounding_shape(adjeacent_characters)
            plate_img, pts, cropped_shape = crop_plate(resized, shape, 0.05)

            if plate_img.shape[0] > 2 and plate_img.shape[1] > 4:
                plate_img = straighten_plate(cropped_shape, plate_img)
                cv2.imshow("Plate | %s" % i, plate_img)

            text = to_text(plate_img)
            cv2.drawContours(resized, [pts], -1, (0, 255, 0), 2)
    if args['video']:
        resized = resized[...,::-1]
    cv2.imshow("Orig | %s" % i, resize(resized, im_show_scale))
    if text != "":
        print(text)

if args['video']:
    source = VideoFileClip(args['video'], fps_source="fps")
    duration = source.duration
    print("Duration: " + str(duration))
    time_frame = 0.2
    time_clip = 0


if args['video']:
    image_in_scale = 0.7
    im_show_scale = 1
    show_processed(source.get_frame(time_clip), args["video"])
    show = True
    while show:
        key = cv2.waitKey(0)
        if not key == -1:
            toggle_images(key)
            show_processed(source.get_frame(time_clip), args["video"])

elif args["image"]:
    img = cv2.imread(args["image"])
    ratio = 1920.0/float(img.shape[1])
    image_in_scale = 0.7
    im_show_scale = ratio*0.6
    show_processed(img, args["image"])
    cv2.waitKey(0)

elif args["multiple"]:
    folder_path = args["multiple"]
    onlyfiles = [f for f in sorted(listdir(folder_path)) if isfile(join(folder_path, f))]
    for file in onlyfiles:
        img = cv2.imread("%s/%s" % (args["multiple"], file))
        ratio = 1920.0/float(img.shape[1])
        image_in_scale = 0.6
        im_show_scale = ratio*0.6
        show_processed(img, file)
    cv2.waitKey(0)
