import os
from xml.etree import ElementTree as ET
import string

"""
    Converts the RWTH dataset into the form we used to train the GAN
"""

with open(
    "../Benchmark/research-seq2seq-HTR/RWTH_partition/RWTH.iam_word_gt_final.train.thresh", "r"
) as r:
    RWTH = r.read()
wid = dict()
for i in os.listdir("XML"):
    root = ET.parse(f"XML/{i}").getroot()
    wid[i.split(".")[0]] = root.attrib["writer-id"]

with open(
    "../Benchmark/research-seq2seq-HTR/RWTH_partition/RWTH.iam_word_gt_final.valid.thresh", "r"
) as r:
    RWTH2 = r.read()
with open(
    "../Benchmark/research-seq2seq-HTR/RWTH_partition/RWTH.iam_word_gt_final.test.thresh", "r"
) as r:
    RWTH3 = r.read()
# remove newline
RWTH = []
# RWTH = RWTH.split("\n")[:-1]
# RWTH.extend(RWTH2.split("\n")[:-1])
RWTH.extend(RWTH3.split("\n")[:-1])

images = [x.split(",")[0] for x in RWTH]
words = [x.split(" ")[1] for x in RWTH]
writer_ids = [wid["-".join(image_name.split("-")[:2])] for image_name in images]
with open("RWTH_PARAGRAPH.txt", "w+") as w:
    for i in range(len(writer_ids)):
        skip = False
        if len(words[i]) >= 11:
            continue
        for char in words[i]:
            if char not in string.ascii_lowercase + string.ascii_uppercase:
                skip = True
        if skip:
            continue
        line = f"{writer_ids[i]},{images[i]} {words[i]}\n"
        print(line, "")
        w.write(line)
