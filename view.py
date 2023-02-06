import math
import cv2
import numpy as np
import graph



def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class Window:
    def __init__(self, path: str):
        self.main = cv2.imread(path)
        self.network_photo = cv2.resize(self.main, (30, 30))
        self.main = cv2.resize(self.network_photo, (300, 300))

        self.view = None
        self.cutted_image = None

        self.foreground_pixels = []
        self.background_pixels = []

        self.point_radius = 10
        self.foreground_pixel_color = (0, 255, 0)
        self.background_pixel_color = (0, 0, 255)

        self.mouseX = None
        self.mouseY = None

        #flags
        self.is_during_cut = False
        self.is_after_cut = False

    def construct_photo(self):
        self.view = self.main.copy()
        for (x, y) in self.foreground_pixels:
            cv2.circle(self.view, (x, y), self.point_radius, self.foreground_pixel_color, -1)

        for (x, y) in self.background_pixels:
            cv2.circle(self.view, (x, y), self.point_radius, self.background_pixel_color, -1)




    def add_ground_pixel(self, event, x, y, flags, param):
        self.mouseX, self.mouseY = x, y
        if self.is_after_cut:
            return
        elif event == cv2.EVENT_LBUTTONDOWN:
            for (x, y) in self.foreground_pixels:
                if distance((x,y), (self.mouseX, self.mouseY)) < self.point_radius:
                    self.foreground_pixels.remove((x, y, ))
                    return
            self.foreground_pixels.append((self.mouseX, self.mouseY, ))

        elif event == cv2.EVENT_RBUTTONDOWN:
            for (x, y) in self.background_pixels:
                if distance((x, y), (self.mouseX, self.mouseY)) < self.point_radius:
                    self.background_pixels.remove((x, y,))
                    return
            self.background_pixels.append((self.mouseX, self.mouseY,))
        else:
            pass

    def cut(self):
        self.is_after_cut = True

        # reshape selected pixels from 300x300 to 30x30
        reshaped_background_pixels = []
        for i in range(len(self.background_pixels)):
            pixel = self.background_pixels[i]
            reshaped_background_pixels.append((pixel[0] // 10, pixel[1] // 10, ))

        reshaped_foreground_pixels = []
        for i in range(len(self.foreground_pixels)):
            pixel = self.foreground_pixels[i]
            reshaped_foreground_pixels.append((pixel[0] // 10, pixel[1] // 10,))


        net = graph.Image(self.network_photo)
        net.do_cut(reshaped_foreground_pixels, reshaped_background_pixels)
        background = net.get_background()

        # make background black
        self.cutted_image = self.network_photo.copy()
        for (x, y) in background:
            self.cutted_image[x][y] = np.array([0, 0, 0])

        self.cutted_image = cv2.resize(self.cutted_image, (300, 300))
        self.is_after_cut = True

    def show(self):

        cv2.namedWindow("image segmentation")
        cv2.setMouseCallback("image segmentation", self.add_ground_pixel)
        while True:
            k = cv2.waitKey(20) & 0xFF

            if not self.is_after_cut:
                self.construct_photo()
                cv2.imshow("image segmentation", self.view)

                if k == ord('d'):
                    self.foreground_pixels = []
                    self.background_pixels = []

                elif k == ord('c'):
                    print("cut")
                    self.cut()




            else:
                cv2.imshow("image segmentation", self.cutted_image)
                if k == ord('r'):
                    self.is_after_cut = False

                elif k == ord('s'):
                    cv2.imwrite("output.jpg", self.cutted_image)


