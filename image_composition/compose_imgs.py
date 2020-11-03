import numpy as np
import random
import json
from image_composition.image_composer import ImageComposer

SEED = 43

def compose_images(config):
    cg = ImageComposer(config)
    for i in range(config["num_imgs"]):
        image_composition, polygon_list = cg.generate_img_composition()
        if config["debug"]:
            cg.show_img_composition(image_composition)
        cg.save_datapoint(image_composition, polygon_list)

if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    collage_config_file = "configs/image_composition.json"
    config = json.load(open(collage_config_file))

