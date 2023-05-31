import os

#path = "/home/jp/Documents/Wallpapers"

class Enumerator():
    def __init__(self, path) -> None:
        files = os.listdir(path)
        self.image_files = []

        for file in files:
            if file.endswith('.jpg') or file.endswith('png') or file.endswith('.jpeg'):
                self.image_files.append(file)

    def get_image_paths(self):
        return self.image_files