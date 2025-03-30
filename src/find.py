from pyexpat import features
from skimage.transform import pyramid_gaussian

from src import *
from src.feture import HOGGenerator

class Find:
    def __init__(self, model : SVC = None, window_size = (64,64), block_size=(16, 16),
                 step_size : int = 8, scale_factor : float = 1.4, detection_threshold : float = 0.85):
        self.model = model
        self.window_size = window_size
        self.block_size = block_size
        self.step_size = step_size
        self.scale_factor = scale_factor
        self.detection_threshold = detection_threshold

        self.features = HOGGenerator(win_size=window_size, block_size=block_size)

    def sliding_window(self, image):
        for y in range(0, image.shape[0] - self.window_size[1] + 1, self.step_size):
            for x in range(0, image.shape[1] - self.window_size[0] + 1, self.step_size):
                yield x, y, image[y:y + self.window_size[1], x:x + self.window_size[0]]

    def find(self, image, detected_classes = ['person']):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for resized in tuple(pyramid_gaussian(gray, downscale=self.scale_factor, channel_axis=-1)):
            scale_ratio = image.shape[1] / float(resized.shape[1])

            for (x, y, window) in self.sliding_window(resized):
                if window.shape[0] != self.window_size[1] or window.shape[1] != self.window_size[0]:
                    continue

                if window.dtype != np.uint8:
                    window = (window * 255).astype(np.uint8)

                features = self.features.generate_hog(window)
                features = features.reshape(1, -1)

                pred = self.model.predict(features)
                prob = self.model.predict_proba(features)[0]


