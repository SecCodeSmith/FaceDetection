from skimage.exposure import exposure

from src import *
class HOGGenerator:
    def __init__(self, win_size=(64, 64), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), n_bins:int=9,
                 deriv_aperture:int=1, win_sigma:float=4., histogram_norm_type=0, l2_hys_threshold:float=2.0000000000000001e-01,
                 gama_correction=0, n_levels:int=64):

        self.last_hog = None
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins, deriv_aperture,
                                     win_sigma, histogram_norm_type, l2_hys_threshold, gama_correction, n_levels)

    def generate_hog(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        self.last_hog = self.hog.compute(image)
        return self.last_hog

    def print_hog(self):
        if self.last_hog is None:
            print("No HOG")
            return
        hog_image_rescaled = exposure.rescale_intensity(self.last_hog, in_range=(0, 10))

        plt.imshow(hog_image_rescaled)
        plt.show()
