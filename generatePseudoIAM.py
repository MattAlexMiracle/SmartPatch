import cv2
import Levenshtein as Lev
import random
import numpy as np
import torch
from network_tro import ConTranModel
from load_data import (
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_WRITERS,
    letter2index,
    tokens,
    num_tokens,
    OUTPUT_MAX_LEN,
    index2letter,
)
from modules_tro import normalize
import os
import argparse
import glob
from collections import defaultdict

parser = argparse.ArgumentParser(
    description="seq2seq net", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--patch_loss",
    dest="patch_loss",
    action="store_true",
    help="uses standard patch loss",
    default=False,
)
parser.add_argument(
    "--smart_patch_loss",
    dest="smart_patch_loss",
    action="store_true",
    help="uses smart patch loss",
    default=False,
)
parser.add_argument(
    "--character_patch_loss",
    dest="character_patch_loss",
    action="store_true",
    help="uses character patch loss",
    default=False,
)
parser.add_argument(
    "--writer_patch_loss",
    dest="writer_patch_loss",
    action="store_true",
    help="uses character patch loss",
    default=False,
)

parser.add_argument("save_weights", type=str, help="location for saving/loading weights")
parser.add_argument("image_folder", type=str, help="location for saving/loading images")


# parser.add_argument("full_loss", type=bool, help="uses the  patch loss")
args = parser.parse_args()
USE_PATCH_GAN = args.patch_loss
USE_SMART_PATCH_GAN = args.smart_patch_loss
USE_CHARACTER_PATCH_GAN = args.character_patch_loss
USE_WRITER_PATCH_GAN = args.writer_patch_loss

USE_FULL_GAN = True  #

gan_type = None
if USE_PATCH_GAN and USE_SMART_PATCH_GAN:
    print("You have to choose either normal OR smart patch gan (or neither)")
    exit(-1)
if USE_PATCH_GAN:
    print("Using Normal patch gan")
    gan_type = "PATCH_GAN"
elif USE_SMART_PATCH_GAN:
    print("Using Smart patch gan")
    gan_type = "SMART_PATCH_GAN"
elif USE_CHARACTER_PATCH_GAN:
    print("Using Character patch gan")
    gan_type = "CHARACTER_PATCH_GAN"
elif USE_WRITER_PATCH_GAN:
    print("Using Writer patch gan")
    gan_type = "WRITER_PATCH_GAN"
else:
    print("Not using partial discriminator")


"""Take turns to open the comments below to run 4 scenario experiments"""

folder = args.image_folder + "/"
weights_folder = args.save_weights + "/"
img_base = "dataset/"
target_file = "Groundtruth/RWTH_PARAGRAPH.txt"  # "Groundtruth/gan.iam.tr_va.gt.filter27"


"""data preparation"""
data_dict = dict()
with open(target_file, "r") as _f:
    data = _f.readlines()
    names = [i.split(" ")[0] for i in data]
    names = [i.split(",") for i in names]
    # get text and remove newline
    texts = [i.split(" ")[1][:-1] for i in data]
data_dict = defaultdict(list)
word_dict = defaultdict(list)
for (wid, index), text in zip(names, texts):
    data_dict[wid].append(index)
    word_dict[wid].append(text)
if not os.path.exists(folder):
    os.makedirs(folder)

gpu = torch.device("cuda")


def test_writer(wid, words, model_file):
    def read_image(file_name, thresh=None):
        url = img_base + file_name + ".png"
        img = cv2.imread(url, 0)
        if img is None and os.path.exists(url):
            # image is present but corrupted
            return False
        if thresh:
            # img[img>thresh] = 255
            pass

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(
            img, (int(img.shape[1] * rate) + 1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC,
        )  # INTER_AREA con error
        img = img / 255.0  # 0-255 -> 0-1

        img = 1.0 - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype="float32")
            outImg[:, :img_width] = img
        outImg = outImg.astype("float32")

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal

    def label_padding(labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        new_label_len.append(len(ll) + 2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
        num = OUTPUT_MAX_LEN - len(ll)
        if not num == 0:
            ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    """data preparation"""
    imgs = list(
        filter(lambda x: isinstance(x, np.ndarray), [read_image(i) for i in data_dict[wid]])
    )

    final_imgs = imgs
    if len(final_imgs) < 50:
        while len(final_imgs) < 50:
            num_cp = 50 - len(final_imgs)
            final_imgs = final_imgs + imgs[:num_cp]
    print(max([len(x) for x in word_dict[wid]]))
    labels = torch.from_numpy(
        np.stack([np.array(label_padding(label, num_tokens)) for label in word_dict[wid]])
    ).to(gpu)
    """model loading"""
    model = ConTranModel(NUM_WRITERS, 0, True, gan_type=gan_type, USE_FULL_GAN=USE_FULL_GAN,).to(
        gpu
    )
    print("Loading " + model_file)
    model.load_state_dict(torch.load(model_file))  # load
    print("Model loaded")
    model.eval()
    num = 0

    with torch.no_grad():
        for label, filename in zip(labels, data_dict[wid]):
            # we chose 50 random writer_ids. This is recomputed every time,
            # to get different outputs if a writer wrote the same word multiple times
            imgs = [final_imgs[x] for x in np.random.choice(len(final_imgs), 50, replace=False)]
            assert len(imgs) == 50
            imgs = torch.from_numpy(np.array(imgs)).unsqueeze(0).to(gpu)  # 1,50,64,216
            f_xs = model.gen.enc_image(imgs)
            label = label.unsqueeze(0)
            f_xt, f_embed = model.gen.enc_text(label, f_xs.shape)
            f_mix = model.gen.mix(f_xs, f_embed)
            xg = model.gen.decode(f_mix, f_xt)
            xg = xg.cpu().numpy().squeeze()
            xg = normalize(xg)
            xg = crop_whitespace(xg)
            xg = 255 - xg
            # _, mask = cv2.threshold(xg, 0, 255, cv2.THRESH_OTSU)
            # xg[mask < 1] = 0
            cv2.imwrite(folder + "/" + filename + ".png", xg)


def crop_whitespace(img):
    """crops out whitespace of generated images"""
    ret, threshholded = cv2.threshold(img, 172, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(threshholded)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y : y + h, x : x + w]
    return rect


if __name__ == "__main__":
    with open(target_file, "r") as _f:
        data = _f.readlines()
    # split into (writerID, text)
    wids = sorted([(i.split(",")[0], i.split()[-1]) for i in data])
    dd = defaultdict(list)
    for wid, text in wids:
        dd[wid].append(text)
    models = glob.glob(weights_folder + "*.model")
    latest_model = [int(m.split(".")[0].split("-")[1]) for m in models] + [0]
    CurriculumModelID = max(latest_model)
    wids, wordList = zip(*dd.items())
    for wid, words in zip(wids, wordList):
        # NOTE that if "word" is in the wordList multiple times, we generate it multiples times
        # print("wid-word pairs", wid, words)
        # test_writer(wid, 'save_weights/<your best model>')
        test_writer(wid, words, f"{weights_folder}/contran-{CurriculumModelID}.model")
