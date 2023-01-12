import sys
from imageio import imread
from skimage.color import rgb2gray
import ImageEditor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import AlternativeMembrane


def F(d):
    return 1 / d ** 3

def main():
    alternative_membrane = False

    # Sanity Check of the given arguments.
    if len(sys.argv[1:]) != 5:
        print("Check number of arguments. Should be 5:\n\
            Source image address\n\
            Target image address\n\
            Binary mask address\n\
            Offset coordinate\n\
            A flag indicating weather to perform gradients mixing")
        exit()

    source_path = sys.argv[1:][0]
    target_path = sys.argv[1:][1]
    mask_path = sys.argv[1:][2]
    offset = (int(sys.argv[1:][3][1]), int(sys.argv[1:][3][3]))
    mixing = sys.argv[1:][4]

    # Converting images addresses to numpy image matrices.
    source = mpimg.imread(source_path) / 255
    target = mpimg.imread(target_path) / 255
    mask = rgb2gray(imread(mask_path))
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    if not alternative_membrane:
        # Editing the images.
        result = ImageEditor.PoissonSeamlessCloning(source, target, mask, offset, mixing == 'True')
        plt.imshow(result)
        plt.show()

    else:
        result = AlternativeMembrane.AlternativeMembraneSeamlessCloning(source, target, mask, offset, F)
        plt.imshow(result)
        plt.show()

if __name__ == "__main__":
    main()