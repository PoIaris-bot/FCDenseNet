import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def augmenter(image, segment_image):
    segment_image = SegmentationMapsOnImage(segment_image, shape=image.shape)
    flip = iaa.SomeOf(
        (1, 2), [
            iaa.Fliplr(),
            iaa.Flipud(),
        ]
    )

    affine = iaa.SomeOf(
        (1, 2), [
            iaa.Affine(scale=(0.5, 1)),
            iaa.Affine(rotate=(-45, 45)),
        ]
    )

    blur = iaa.OneOf([
        iaa.GaussianBlur(),
        iaa.AverageBlur(k=((2, 11), (2, 11))),
        iaa.Sharpen(),
    ])

    noise = iaa.SomeOf(
        (1, 2), [
            iaa.AdditiveGaussianNoise(),
            iaa.CoarseDropout(p=(0, 0.2), size_percent=0.2),
        ]
    )

    colorspace = iaa.OneOf([
        iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="BGR",
            children=iaa.Multiply((0.5, 1.5), per_channel=0.5)
        ),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
    ])

    augment = iaa.Sequential([
        flip if np.random.rand() < 0.5 else iaa.Noop(),
        affine if np.random.rand() < 0.5 else iaa.Noop(),
        blur if np.random.rand() < 0.5 else iaa.Noop(),
        noise if np.random.rand() < 0.5 else iaa.Noop(),
        colorspace if np.random.rand() < 0.5 else iaa.Noop()
    ], random_order=True)

    image, segment_image = augment(image=image, segmentation_maps=segment_image)
    return image, segment_image.get_arr()