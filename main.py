import view
import sys

pic_path = sys.argv[1]
def main(path: str):
    window = view.Window(path)
    window.show()

if __name__ == "__main__":
    main(pic_path)