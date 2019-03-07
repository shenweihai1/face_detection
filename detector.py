from __future__ import print_function, division
import face_alignment
from skimage import io
import numpy as np
from PIL import Image
import warnings
from os.path import abspath, dirname, join
BASE_PATH = abspath(dirname(abspath(__file__)))
warnings.filterwarnings("ignore")
STATUS = {
    1: join(BASE_PATH, 'emoji/h.jpg'),  # happy
    2: join(BASE_PATH, 'emoji/s.jpg'),  # sad
    3: join(BASE_PATH, 'emoji/n.jpg'),  # neutral
}


class Detector(object):
    def __init__(self, device="cpu"):
        # download the weights model when first time
        self.face = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    def replacement(self, input_image, out_image, status):
        if status not in STATUS:
            return False, "status is valid"
        try:
            emotion_image = STATUS[status]

            X = io.imread(input_image)
            landmarks = self.face.get_landmarks_from_image(X)[-1]
            box_width = min(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]), np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]))

            input_image_rev = Image.open(input_image)
            emotion_image_rev = Image.open(emotion_image)
            wc, hc = emotion_image_rev.size
            if wc != hc:
                raise Exception(emotion_image + " must have the same width and height")

            zoom = box_width / (wc + 0.0)
            emotion_image_rev_scale = emotion_image_rev.resize((int(wc * zoom), int(wc*zoom)))
            offset = (int(np.mean(landmarks[:, 0]) - 0.5 * (int(wc * zoom))), int(np.mean(landmarks[:, 1]) - 0.5 * (int(wc * zoom))))
            input_image_rev.paste(emotion_image_rev_scale, offset)
            input_image_rev.save(out_image)
        except Exception as ex:
            return False, str(ex)
        return True, ""


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    obj = Detector()

    for img in ["8.jpg"]:
        input_image = join(BASE_PATH, "assets", img)
        out_image = join(BASE_PATH, "assets", "out_" + img)
        status, msg = obj.replacement(input_image, out_image, random.randint(1, 3))

        if status:
            plt.imshow(io.imread(out_image))
            plt.pause(10)
        else:
            print(msg)
