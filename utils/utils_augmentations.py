import config as config
import albumentations as albu


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.PadIfNeeded(config.imagesize, config.imagesize)]
    return albu.Compose(test_transform)


def get_training_augmentation(augmentations="weak"):
    """

    :param augmentations: choose between "weak" (only geometric) and "strong" (spectral)
    :return:
    """
    if augmentations == "strong":
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.OneOf(
                [
                    # albu.Perspective(scale=(0.05, 0.1), p=1),
                    albu.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.3,
                        rotate_limit=69,
                        border_mode=0,
                        p=1,
                    ),
                ],
                p=1,
            ),
            albu.OneOf(
                [
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.5,
            ),
            albu.OneOf(
                [
                    albu.Blur(blur_limit=(3, 5), p=1),  # default blur_limit=(3, 7)
                    albu.MotionBlur(
                        blur_limit=(3, 5), p=1
                    ),  # default blur_limit=(3, 7)
                ],
                p=0.2,
            ),
            albu.OneOf(
                [
                    albu.RandomContrast(limit=0.2, p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.5,
            ),
        ]
    else:
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.3, rotate_limit=69, border_mode=0, p=0.5
            ),
        ]

    return albu.Compose(train_transform)
