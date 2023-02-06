import view

some_pic = "images/3.jpg"
def main(path: str):
    window = view.Window(path)
    window.show()

if __name__ == "__main__":
    main(some_pic)