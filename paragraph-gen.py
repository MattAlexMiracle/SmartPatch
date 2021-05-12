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
import matplotlib.pyplot as plt

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

parser.add_argument(
    "--pure_IAM",
    dest="pure_IAM",
    action="store_true",
    help="generatesPragraphs for Pure IAM",
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


folder = args.image_folder + "/"
weights_folder = args.save_weights + "/"
img_base = "dataset/"
target_file = "Groundtruth/RWTH_GAN_FORMAT.txt"  # "Groundtruth/gan.iam.tr_va.gt.filter27"


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


def test_writer(wid, model, text, nr):
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
    text_transformed = torch.from_numpy(
        np.stack([np.array(label_padding(word, num_tokens)) for word in text])
    ).to(gpu)

    num = 0
    word_images = []
    image_batch = torch.tensor(
        [
            [final_imgs[x] for x in np.random.choice(len(final_imgs), 50, replace=False)]
            for _ in range(len(text))
        ]
    ).to(gpu)
    assert len(image_batch) == len(text)
    with torch.no_grad():
        # for word in text_transformed:
        # we chose 50 random writer_ids. This is recomputed every time,
        # to get different outputs if a writer wrote the same word multiple times
        # imgs = [final_imgs[x] for x in np.random.choice(len(final_imgs), 50, replace=False)]
        # assert len(imgs) == 50
        # imgs = torch.from_numpy(np.array(imgs)).unsqueeze(0).to(gpu)  # 1,50,64,216
        f_xs = model.gen.enc_image(image_batch)
        word = text_transformed
        f_xt, f_embed = model.gen.enc_text(word, f_xs.shape)
        f_mix = model.gen.mix(f_xs, f_embed)
        xg = model.gen.decode(f_mix, f_xt)
        xg = xg.cpu().numpy().squeeze()
        xg = [255 - crop_whitespace(normalize(x)) for x in xg]
        # _, mask = cv2.threshold(xg, 0, 255, cv2.THRESH_OTSU)
        # xg[mask < 1] = 0
        # cv2.imwrite(folder + "/" + filename + ".png", xg)
        word_images = xg
    canvas = place_words_on_canvas(word_images)
    # plt.imshow(canvas,cmap='gray', vmin=0, vmax=255)
    # plt.show()
    cv2.imwrite(folder + "/" + str(wid) + "-" + str(nr) + ".png", canvas)


def place_words_on_canvas(words):
    NR_Lines, NR_WORDS_IN_LINE = 7, 7
    empty_space = 20
    paragraph = (
        np.ones(
            ((NR_Lines) * IMG_HEIGHT, empty_space + NR_WORDS_IN_LINE * (IMG_WIDTH + empty_space))
        )
        * 255
    )
    for lineN in range(NR_Lines):
        offset = empty_space
        for wordN in range(NR_WORDS_IN_LINE):
            assert offset < NR_WORDS_IN_LINE * (IMG_WIDTH + empty_space)
            idx = wordN + NR_WORDS_IN_LINE * lineN
            paragraph[
                IMG_HEIGHT * lineN : IMG_HEIGHT * (lineN + 1), offset : offset + words[idx].shape[1]
            ] = words[idx]
            offset += words[idx].shape[1] + empty_space
    return paragraph


def crop_whitespace(img):
    """crops out whitespace of generated images"""
    ret, threshholded = cv2.threshold(img, 172, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(threshholded)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[:, x : x + w]
    return rect


def gen_IAM(dd, nr_paragraph):
    for key in dd.keys():
        imgs_loaded = [cv2.imread("dataset/" + name, cv2.COLOR_BGR2GRAY) for name in dd[key]]
        for nr in range(nr_paragraph):
            idx = np.random.choice(len(dd[key]), replace=True, size=7 * 7)
            print(dd[key][0])
            words = [imgs_loaded[x] for x in idx]

            # scale the words based on the maximal height of the  writer's words,
            # then pad them with whitespace to IMG_HEIGHT. This keeps the relative sizes between word-images
            m = max([w.shape[0] for w in words])
            words = [cv2.resize(w, dsize=(-1, -1), fx=64 / m, fy=64 / m) for w in words]
            # rescale the maximum image to be at most IMG_WIDTH long
            maxlen = max([w.shape[1] for w in words])
            print(maxlen)
            if maxlen >= IMG_WIDTH:
                ratio = IMG_WIDTH / maxlen
                words = [cv2.resize(w, dsize=(-1, -1), fx=ratio, fy=ratio) for w in words]
            for idx, w in enumerate(words):
                empty = np.ones((IMG_HEIGHT, w.shape[1])) * 255
                empty[0 : w.shape[0], 0 : w.shape[1]] = w
                words[idx] = empty
                # plt.imshow(empty)
                # plt.show()
                # print(empty.shape)

            canvas = place_words_on_canvas(words)
            cv2.imwrite(folder + "/" + str(key) + "-" + str(nr) + ".png", canvas)


if __name__ == "__main__":
    with open(target_file, "r") as _f:
        data = _f.readlines()
    # split into (writerID, text)
    wids = sorted([(i.split(",")[0], i.split(",")[1].split(" ")[0]) for i in data])
    dd = defaultdict(list)
    for wid, file in wids:
        dd[wid].append(file + ".png")
    if args.pure_IAM:
        gen_IAM(dd, 10)
        exit(0)
    models = glob.glob(weights_folder + "*.model")
    latest_model = [int(m.split(".")[0].split("-")[1]) for m in models] + [0]
    CurriculumModelID = max(latest_model)
    wids, wordList = zip(*dd.items())
    with open("Benchmark/paragraphs.txt", "r") as reader:
        file = reader.read().split("\n")
    for wid in wids:
        for nr, paragraph in enumerate(file):
            print(nr)
            text = paragraph.split(" ")
            """model loading"""
            model = ConTranModel(
                NUM_WRITERS, 0, True, gan_type=gan_type, USE_FULL_GAN=USE_FULL_GAN,
            ).to(gpu)
            print("Loading " + f"{weights_folder}/contran-{CurriculumModelID}.model")
            model.load_state_dict(
                torch.load(f"{weights_folder}/contran-{CurriculumModelID}.model")
            )  # load
            print("Model loaded")
            model.eval()
            test_writer(wid, model, text, nr)
