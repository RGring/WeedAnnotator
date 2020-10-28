class Annotation:
    def __init__(self, image_location, img_width, img_height, bb_list, polygon_list):
        self.image_name = image_location
        self.img_width = img_width
        self.img_height = img_height
        self.bb_list = bb_list
        self.polygon_list = polygon_list

    def get_image_name(self):
        return self.image_name

    def get_img_width(self):
        return self.img_width

    def get_img_height(self):
        return self.img_height

    def get_bounding_boxes(self):
        return self.bb_list

    def get_polygons(self):
        return self.polygon_list
