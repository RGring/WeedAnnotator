from PIL import Image
import imagehash
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class DuplicateRemover:
    def __init__(self, dirname, hash_size=7):
        self.dirname = dirname
        self.hash_size = hash_size
        self._pressed_key = ""

    def _key_press(self, event):
        self._pressed_key = event.key
        plt.close()

    def find_duplicates(self):
        """
        Find and Delete Duplicates
        """

        fnames = os.listdir(self.dirname)
        hashes = {}
        duplicates = []
        print("Finding Duplicates Now!\n")
        for image in fnames:
            with Image.open(os.path.join(self.dirname, image)) as img:
                temp_hash = imagehash.average_hash(img, self.hash_size)
                if temp_hash in hashes:
                    print("Duplicate {} \nfound for Image {}!\n".format(image, hashes[temp_hash]))
                    # fig, axes = plt.subplots(2, 1)
                    # fig.canvas.mpl_connect('key_press_event', self._key_press)
                    # img1 = cv2.imread(os.path.join(self.dirname, image))
                    # img2 = cv2.imread(os.path.join(self.dirname, hashes[temp_hash]))
                    # axes[0].imshow(img1)
                    # axes[1].imshow(img2)
                    # plt.show()
                    #
                    # if self._pressed_key == "enter":
                    #     pass
                    # elif self._pressed_key == "d":
                    #     print(f"Removing: {image}")
                    #     os.remove(os.path.join(self.dirname, image))
                    # elif self._pressed_key == "del":
                    #     break
                    os.remove(os.path.join(self.dirname, image))
                    #duplicates.append(image)
                else:
                    hashes[temp_hash] = image

        if len(duplicates) != 0:
            a = input("Do you want to delete these {} Images? Press Y or N:  ".format(len(duplicates)))
            space_saved = 0
            if (a.strip().lower() == "y"):
                for duplicate in duplicates:
                    space_saved += os.path.getsize(os.path.join(self.dirname, duplicate))

                    os.remove(os.path.join(self.dirname, duplicate))
                    print("{} Deleted Succesfully!".format(duplicate))

                print("\n\nYou saved {} mb of Space!".format(round(space_saved / 1000000), 2))
            else:
                print("Thank you for Using Duplicate Remover")
        else:
            print("No Duplicates Found :(")

    def find_similar(self, location, similarity=80):
        fnames = os.listdir(self.dirname)
        threshold = 1 - similarity / 100
        diff_limit = int(threshold * (self.hash_size ** 2))

        with Image.open(location) as img:
            hash1 = imagehash.average_hash(img, self.hash_size).hash

        print("Finding Similar Images to {} Now!\n".format(location))
        for image in fnames:
            with Image.open(os.path.join(self.dirname, image)) as img:
                hash2 = imagehash.average_hash(img, self.hash_size).hash

                if np.count_nonzero(hash1 != hash2) <= diff_limit:
                    print("{} image found {}% similar to {}".format(image, similarity, location))

if __name__ == "__main__":
    dirname = "images"

    # Remove Duplicates
    dr = DuplicateRemover("/home/rog/data/sugar_beet/rbg_wo_duplicates")
    dr.find_duplicates()

    # Find Similar Images
    # dr.find_similar("IMG-20110704-00007.jpg",70)