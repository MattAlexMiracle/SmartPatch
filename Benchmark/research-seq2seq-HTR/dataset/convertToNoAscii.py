import string


def convertToAscii(pathIn, pathOut):
    with open(pathIn, "r") as reader:
        lines = reader.read().split("\n")
    with open(pathOut, "w+") as writer:
        for line in lines[:-1]:
            cont = False
            word = " ".join(line.split(" ")[1:])

            if len(word) >= 11:
                continue
            for j in word:
                if not (j in string.ascii_uppercase + string.ascii_lowercase):
                    cont = True
            if cont:
                continue
            print(word)
            writer.write(line.strip() + "\n")


convertToAscii("RWTH.iam_word_gt_final.train.thresh", "RWTH.iam_word_gt_final.ascii.train.thresh")
convertToAscii("RWTH.iam_word_gt_final.test.thresh", "RWTH.iam_word_gt_final.ascii.test.thresh")
convertToAscii("RWTH.iam_word_gt_final.valid.thresh", "RWTH.iam_word_gt_final.ascii.valid.thresh")
