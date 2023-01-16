import graph
import cv2

def test():
    img = cv2.imread("images/5.jpg")
    img = graph.cut(img, (70, 70), (0, 0))
    cv2.imshow("cut", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()