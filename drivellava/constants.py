import itertools
import os
from glob import glob


class Indexable(object):
    def __init__(self, it):
        self.it = iter(it)

    def __iter__(self):
        return self.it

    def __getitem__(self, index):
        try:
            return next(itertools.islice(self.it, index, index + 1))
        except TypeError:
            return list(
                itertools.islice(self.it, index.start, index.stop, index.step)
            )


COMMAVQ_DIR = os.path.expanduser("~/Datasets/commavq")

# List of all the videos
ENCODED_VIDEOS_ALL = glob(os.path.join(COMMAVQ_DIR, "*", "*.npy"))
ENCODED_VIDEOS_ALL = [x for x in ENCODED_VIDEOS_ALL if os.path.isfile(x)]
# ENCODED_VIDEOS_ALL = sorted(ENCODED_VIDEOS_ALL)

ENCODED_POSE_ALL = glob(os.path.join(COMMAVQ_DIR, "pose_data_*", "*.npy"))
ENCODED_POSE_ALL = [x for x in ENCODED_POSE_ALL if os.path.isfile(x)]

# List of all the encoded videos
ENCODED_VIDEOS = glob(os.path.join(COMMAVQ_DIR, "data_*_to_*", "*.npy"))
ENCODED_VIDEOS = [x for x in ENCODED_VIDEOS if os.path.isfile(x)]

ENCODED_POSE = glob(os.path.join(COMMAVQ_DIR, "pose_data_*_to_*", "*.npy"))
ENCODED_POSE = [x for x in ENCODED_POSE if os.path.isfile(x)]

VAL_ENCODED_VIDEOS = glob(os.path.join(COMMAVQ_DIR, "val", "*.npy"))
VAL_ENCODED_VIDEOS = [x for x in VAL_ENCODED_VIDEOS if os.path.isfile(x)]

VAL_ENCODED_POSE = glob(os.path.join(COMMAVQ_DIR, "pose_val", "*.npy"))
VAL_ENCODED_POSE = [x for x in VAL_ENCODED_POSE if os.path.isfile(x)]

DECODED_IMGS = {
    x: (
        list(
            os.path.join(
                x.replace("data_", "img_data_").replace(".npy", ""), f"{j}.png"
            )
            for j in range(1200)
        )
    )
    for x in ENCODED_VIDEOS_ALL
}

VAL_DECODED_IMGS = {
    x: (
        list(
            os.path.join(
                x.replace("val", "img_val").replace(".npy", ""), f"{j}.png"
            )
            for j in range(1200)
        )
    )
    for x in VAL_ENCODED_VIDEOS
}

DECODED_IMGS_ALL = {**DECODED_IMGS, **VAL_DECODED_IMGS}

# Find available images

DECODED_IMGS_AVAILABLE = {
    k: [x for x in v if os.path.isfile(x)] for k, v in DECODED_IMGS_ALL.items()
}

VAL_DECODED_IMGS_AVAILABLE = {
    k: [x for x in v if os.path.isfile(x)] for k, v in VAL_DECODED_IMGS.items()
}

DECODED_IMGS_ALL_AVAILABLE = {
    **DECODED_IMGS_AVAILABLE,
    **VAL_DECODED_IMGS_AVAILABLE,
}


COMMAVQ_GPT2M_DIR = os.path.expanduser("~/Datasets/commavq-gpt2m")

ENCODER_ONNX_PATH = os.path.join(COMMAVQ_GPT2M_DIR, "encoder.onnx")

DECODER_ONNX_PATH = os.path.join(COMMAVQ_GPT2M_DIR, "decoder.onnx")

assert os.path.isfile(ENCODER_ONNX_PATH)
assert os.path.isfile(DECODER_ONNX_PATH)

assert len(ENCODED_VIDEOS) > 0
assert len(VAL_ENCODED_VIDEOS) > 0
assert len(ENCODED_POSE) > 0
assert len(VAL_ENCODED_VIDEOS) > 0

# print('DECODED_IMGS', DECODED_IMGS)
