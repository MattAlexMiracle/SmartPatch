import os
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from load_data import NUM_WRITERS
from network_tro import ConTranModel
import modules_tro
from load_data import loadData as load_data_func
from loss_tro import CER
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="seq2seq net", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("start_epoch", type=int, help="load saved weights from which epoch")
parser.add_argument(
    "--patch_loss", dest="patch_loss", action="store_true", help="uses standard patch loss", default=False,
)
parser.add_argument(
    "--smart_patch_loss", dest="smart_patch_loss", action="store_true", help="uses smart patch loss", default=False,
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
parser.add_argument("image_folder", type=str, help="location for saving images")


args = parser.parse_args()

save_weights_path = args.save_weights + "/"
image_folder = args.image_folder + "/"

gpu = torch.device("cuda")

OOV = True

NUM_THREAD = 2

EARLY_STOP_EPOCH = 500  # experimentally determined by looking at the worst-case behavior in the first 2000 epochs
EVAL_EPOCH = 20
MODEL_SAVE_EPOCH = 200
show_iter_num = 500
# 24 hours is the maximum cluster runtime, so save beforehand
max_time = 24 * 60 * 60 - 5 * 60
start_time = time.time()
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 8
lr_dis = 1 * 1e-4
lr_gen = 1 * 1e-4
lr_rec = 1 * 1e-5
lr_cla = 1 * 1e-5

CurriculumModelID = args.start_epoch
USE_PATCH_GAN = args.patch_loss
USE_SMART_PATCH_GAN = args.smart_patch_loss
USE_CHARACTER_PATCH_GAN = args.character_patch_loss
USE_WRITER_PATCH_GAN = args.writer_patch_loss
USE_FULL_GAN = True  #


def all_data_loader():
    data_train, data_test = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(
        data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True,
    )
    return train_loader, test_loader


def sort_batch(batch):
    train_domain = list()
    train_wid = list()
    train_idx = list()
    train_img = list()
    train_img_width = list()
    train_label = list()
    img_xts = list()
    label_xts = list()
    label_xts_swap = list()
    for (domain, wid, idx, img, img_width, label, img_xt, label_xt, label_xt_swap,) in batch:
        if wid >= NUM_WRITERS:
            print("error!")
        train_domain.append(domain)
        train_wid.append(wid)
        train_idx.append(idx)
        train_img.append(img)
        train_img_width.append(img_width)
        train_label.append(label)
        img_xts.append(img_xt)
        label_xts.append(label_xt)
        label_xts_swap.append(label_xt_swap)

    train_domain = np.array(train_domain)
    train_idx = np.array(train_idx)
    train_wid = np.array(train_wid, dtype="int64")
    train_img = np.array(train_img, dtype="float32")
    train_img_width = np.array(train_img_width, dtype="int64")
    train_label = np.array(train_label, dtype="int64")
    img_xts = np.array(img_xts, dtype="float32")
    label_xts = np.array(label_xts, dtype="int64")
    label_xts_swap = np.array(label_xts_swap, dtype="int64")

    train_wid = torch.from_numpy(train_wid)
    train_img = torch.from_numpy(train_img)
    train_img_width = torch.from_numpy(train_img_width)
    train_label = torch.from_numpy(train_label)
    img_xts = torch.from_numpy(img_xts)
    label_xts = torch.from_numpy(label_xts)
    label_xts_swap = torch.from_numpy(label_xts_swap)

    return (
        train_domain,
        train_wid,
        train_idx,
        train_img,
        train_img_width,
        train_label,
        img_xts,
        label_xts,
        label_xts_swap,
    )


def train(
    train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch, tracking_dict_train,
):
    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_cla = list()
    loss_cla_tr = list()
    loss_l1 = list()
    loss_rec = list()
    loss_rec_tr = list()
    loss_dis_tr_full = list()
    loss_dis_full = list()
    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()
    for train_data_list in train_loader:
        """rec update"""
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, "rec_update", cer_tr)
        rec_opt.step()

        """classifier update"""
        l_cla_tr = torch.zeros(1).to(gpu)
        cla_opt.zero_grad()
        l_cla_tr = model(train_data_list, epoch, "cla_update")
        cla_opt.step()

        """dis update"""
        dis_opt.zero_grad()
        l_dis_tr, l_dis_tr_full = model(train_data_list, epoch, "dis_update")
        dis_opt.step()

        """gen update"""
        gen_opt.zero_grad()
        l_total, l_dis, l_dis_full, l_cla, l_l1, l_rec = model(train_data_list, epoch, "gen_update", [cer_te, cer_te2])
        gen_opt.step()

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_cla_tr.append(l_cla_tr.cpu().item())
        loss_l1.append(l_l1.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        loss_rec_tr.append(l_rec_tr.cpu().item())
        loss_dis_tr_full.append(l_dis_tr_full.cpu().item())
        loss_dis_full.append(l_dis_full.cpu().item())

    fl_dis = np.mean(loss_dis)
    fl_dis_tr = np.mean(loss_dis_tr)
    fl_cla = np.mean(loss_cla)
    fl_cla_tr = np.mean(loss_cla_tr)
    fl_l1 = np.mean(loss_l1)
    fl_rec = np.mean(loss_rec)
    fl_rec_tr = np.mean(loss_rec_tr)
    fl_dis_tr_full = np.mean(loss_dis_tr_full)
    fl_dis_full = np.mean(loss_dis_full)

    res_cer_tr = cer_tr.fin()
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print(
        "epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_dis_full=%.2f-%.2f, l_cla=%.2f-%.2f, l_rec=%.2f-%.2f, l1=%.2f, cer=%.2f-%.2f-%.2f, time=%.1f"
        % (
            epoch,
            fl_dis_tr,
            fl_dis,
            fl_dis_tr_full,
            fl_dis_full,
            fl_cla_tr,
            fl_cla,
            fl_rec_tr,
            fl_rec,
            fl_l1,
            res_cer_tr,
            res_cer_te,
            res_cer_te2,
            time.time() - time_s,
        )
    )
    tracking_dict_train["epoch"].append(epoch)
    tracking_dict_train["fl_dis_tr"].append(fl_dis_tr)
    tracking_dict_train["fl_dis"].append(fl_dis)
    tracking_dict_train["fl_dis_tr_full"].append(fl_dis_tr_full)
    tracking_dict_train["fl_dis_full"].append(fl_dis_full)
    tracking_dict_train["fl_cla_tr"].append(fl_cla_tr)
    tracking_dict_train["fl_cla"].append(fl_cla)
    tracking_dict_train["fl_rec_tr"].append(fl_rec_tr)
    tracking_dict_train["fl_rec"].append(fl_rec)
    tracking_dict_train["fl_l1"].append(fl_l1)
    tracking_dict_train["res_cer_tr"].append(res_cer_tr)
    tracking_dict_train["res_cer_te"].append(res_cer_te)
    tracking_dict_train["res_cer_te2"].append(res_cer_te2)

    return res_cer_te + res_cer_te2


def test(test_loader, epoch, modelFile_o_model, tracking_dict_eval):
    if type(modelFile_o_model) == str:
        model = ConTranModel(NUM_WRITERS, show_iter_num, OOV).to(gpu)
        print("Loading " + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model))  # load
    else:
        model = modelFile_o_model
    model.eval()
    loss_dis = list()
    loss_cla = list()
    loss_rec = list()
    loss_dis_full = list()
    time_s = time.time()
    cer_te = CER()
    cer_te2 = CER()
    for test_data_list in test_loader:
        l_dis, l_dis_full, l_cla, l_rec = model(test_data_list, epoch, "eval", [cer_te, cer_te2])

        loss_dis.append(l_dis.cpu().item())
        loss_dis_full.append(l_dis_full.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_rec.append(l_rec.cpu().item())

    fl_dis = np.mean(loss_dis)
    fl_dis_full = np.mean(loss_dis_full)
    fl_cla = np.mean(loss_cla)
    fl_rec = np.mean(loss_rec)

    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print(
        "EVAL: l_dis=%.3f, l_dis_full=%.3f, l_cla=%.3f, l_rec=%.3f, cer=%.2f-%.2f, time=%.1f"
        % (fl_dis, fl_dis_full, fl_cla, fl_rec, res_cer_te, res_cer_te2, time.time() - time_s,)
    )
    tracking_dict_eval["fl_dis"].append(fl_dis)
    tracking_dict_eval["fl_dis_full"].append(fl_dis_full)
    tracking_dict_eval["fl_cla"].append(fl_cla)
    tracking_dict_eval["fl_rec"].append(fl_rec)
    tracking_dict_eval["res_cer_te"].append(res_cer_te)
    tracking_dict_eval["res_cer_te2"].append(res_cer_te2)


def main(train_loader, test_loader, num_writers, gan_type):

    model = ConTranModel(num_writers, show_iter_num, OOV, gan_type=gan_type, USE_FULL_GAN=USE_FULL_GAN).to(gpu)
    tracking_dict_train = {
        "epoch": [],
        "fl_dis_tr": [],
        "fl_dis": [],
        "fl_dis_tr_full": [],
        "fl_dis_full": [],
        "fl_cla_tr": [],
        "fl_cla": [],
        "fl_rec_tr": [],
        "fl_rec": [],
        "fl_l1": [],
        "res_cer_tr": [],
        "res_cer_te": [],
        "res_cer_te2": [],
    }
    tracking_dict_eval = {
        "fl_dis": [],
        "fl_dis_full": [],
        "fl_cla": [],
        "fl_rec": [],
        "res_cer_te": [],
        "res_cer_te2": [],
    }
    if CurriculumModelID > 0:
        model_file = save_weights_path + "contran-" + str(CurriculumModelID) + ".model"
        print("Loading " + model_file)
        model.load_state_dict(torch.load(model_file))  # load
        with open(save_weights_path + "tracking-" + str(CurriculumModelID) + ".pickle", "rb") as reader:
            dicts = pickle.load(reader)
        tracking_dict_eval = dicts["eval"]
        tracking_dict_train = dicts["train"]
        # pretrain_dict = torch.load(model_file)
        # model_dict = model.state_dict()
        # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and not k.startswith('gen.enc_text.fc')}
        # model_dict.update(pretrain_dict)
        # model.load_state_dict(model_dict)
    dis_full_params = []
    if model.dis_full is not None:
        dis_full_params = list(model.dis_full.parameters())
        print("using full and partial discriminator")
    dis_params = list(model.dis.parameters()) + dis_full_params

    gen_params = list(model.gen.parameters())
    rec_params = list(model.rec.parameters())
    cla_params = list(model.cla.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    cla_opt = optim.Adam([p for p in cla_params if p.requires_grad], lr=lr_cla)
    epochs = 50001
    # epochs = 2001
    min_cer = 1e5
    min_idx = 0
    min_count = 0

    for epoch in range(CurriculumModelID, epochs):
        cer = train(
            train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch, tracking_dict_train=tracking_dict_train,
        )

        if epoch % MODEL_SAVE_EPOCH == 0 or time.time() - start_time >= max_time:
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)
            torch.save(model.state_dict(), save_weights_path + "/contran-%d.model" % epoch)
            with open(save_weights_path + "/tracking-%d.pickle" % epoch, "wb") as writer:
                pickle.dump(
                    {"train": tracking_dict_train, "eval": tracking_dict_eval},
                    writer,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            """for key in tracking_dict_train.keys():
                if key != "epoch":
                    plt.plot(tracking_dict_train[key], label=key)
            plt.legend()
            plt.savefig(save_weights_path + "/plot-train-%d.png" % epoch)
            plt.clf()
            for key in tracking_dict_eval.keys():
                plt.plot(tracking_dict_eval[key], label=key)
            plt.legend()
            plt.savefig(save_weights_path + "/plot-eval-%d.png" % epoch)
            plt.clf()"""

        if epoch % EVAL_EPOCH == 0:
            test(test_loader, epoch, model, tracking_dict_eval=tracking_dict_eval)

        if EARLY_STOP_EPOCH is not None:
            if min_cer > cer:
                min_cer = cer
                min_idx = epoch
                min_count = 0
                rm_old_model(min_idx)
            else:
                min_count += 1
            if min_count >= EARLY_STOP_EPOCH:
                if not os.path.exists(save_weights_path):
                    os.makedirs(save_weights_path)
                torch.save(model.state_dict(), save_weights_path + "/contran-%d.model" % epoch)
                with open(save_weights_path + "/tracking-%d.pickle" % epoch, "wb") as writer:
                    pickle.dump(
                        {"train": tracking_dict_train, "eval": tracking_dict_eval},
                        writer,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                """for key in tracking_dict_train.keys():
                    if key != "epoch":
                        plt.plot(tracking_dict_train[key], label=key)
                plt.legend()
                plt.savefig(save_weights_path + "/plot-train-%d.png" % epoch)
                plt.clf()
                for key in tracking_dict_eval.keys():
                    plt.plot(tracking_dict_eval[key], label=key)
                plt.legend()
                plt.savefig(save_weights_path + "/plot-eval-%d.png" % epoch)
                plt.clf()"""
                print("Early stop at %d and the best epoch is %d" % (epoch, min_idx))
                # model_url = save_weights_path + "contran-" + str(min_idx) + ".model"
                # os.system("mv " + model_url + " " + model_url + ".bak")
                # os.system("rm {}contran-*.model".format(save_weights_path))
                break


def rm_old_model(index):
    models = glob.glob(save_weights_path + "*.model")
    # always keep two states as backup.
    index = index - 2 * MODEL_SAVE_EPOCH
    for m in models:
        epoch = int(m.split(".")[0].split("-")[1])
        if epoch < index:
            os.system("rm {}contran-".format(save_weights_path) + str(epoch) + ".model")


if __name__ == "__main__":
    print(time.ctime())
    gan_type = None
    modules_tro.image_folder = image_folder
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

    if CurriculumModelID == 0:
        exec(open("clean.py").read())
    if CurriculumModelID < 0:
        models = glob.glob(save_weights_path + "*.model")
        latest_model = [int(m.split(".")[0].split("-")[1]) for m in models] + [0]
        CurriculumModelID = max(latest_model)
    print("loading model id", CurriculumModelID)

    train_loader, test_loader = all_data_loader()
    main(train_loader, test_loader, NUM_WRITERS, gan_type)
    print(time.ctime())
