import torch
import torch.nn as nn
from load_data import vocab_size, IMG_WIDTH, OUTPUT_MAX_LEN
from modules_tro import (
    GenModel_FC,
    DisModel,
    DisModelPatch,
    DisModelPatchSmart,
    DisModelPatchWithCharacters,
    DisModelPatchWithStyle,
    WriterClaModel,
    RecModel,
    write_image,
)
from loss_tro import recon_criterion, crit, log_softmax
import numpy as np

w_dis = 1.0
w_cla = 1.0
# w_cla = 0.0
w_l1 = 0.0
w_rec = 1.0
# USE_PATCH_GAN = False
# USE_SMART_PATCH_GAN = True
# USE_FULL_GAN = True
N_layers = 3
ADDITIONAL_CAP = False

gpu = torch.device("cuda")


def count_parameters(model):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConTranModel(nn.Module):
    def __init__(
        self, num_writers, show_iter_num, oov, gan_type, USE_FULL_GAN,
    ):
        super(ConTranModel, self).__init__()
        self.gen = GenModel_FC(OUTPUT_MAX_LEN).to(gpu)
        self.cla = WriterClaModel(num_writers).to(gpu)
        self.rec = RecModel(pretrain=False).to(gpu)
        print("Choosing model")
        if gan_type == "PATCH_GAN":
            self.dis = DisModelPatch(n_layers=N_layers, additional_cap=ADDITIONAL_CAP).to(gpu)
            print("PATCH_GAN")
        elif gan_type == "SMART_PATCH_GAN":
            self.dis = DisModelPatchSmart(self.rec, n_layers=N_layers, additional_cap=ADDITIONAL_CAP).to(gpu)
            print("SMART_PATCH_GAN")
        elif gan_type == "CHARACTER_PATCH_GAN":
            self.dis = DisModelPatchWithCharacters(self.rec, n_layers=N_layers, additional_cap=ADDITIONAL_CAP).to(gpu)
            print("CHARACTER_PATCH_GAN")
            """elif gan_type == "WRITER_PATCH_GAN":
                print("WRITER_PATCH_GAN")
                self.dis = DisModelPatchWithStyle(
                    self.rec, n_layers=N_layers, additional_cap=ADDITIONAL_CAP
                    )"""
        else:
            print("NO_PATCH_GAN")
            self.dis = DisModel().to(gpu)
        self.dis_full = None
        if USE_FULL_GAN and not (gan_type == None):
            self.dis_full = DisModel().to(gpu)
        else:
            USE_FULL_GAN = False

        print(
            "parameters:",
            "generator",
            count_parameters(self.gen),
            "writer classifier",
            count_parameters(self.cla),
            "Discriminator",
            count_parameters(self.dis),
            "word recognizer",
            count_parameters(self.rec),
            "full discriminator",
            count_parameters(self.dis_full),
        )
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        self.oov = oov
        self.gan_type = gan_type
        self.USE_FULL_GAN = USE_FULL_GAN

    def forward(self, train_data_list, epoch, mode, cer_func=None):
        (tr_domain, tr_wid, tr_idx, tr_img, tr_img_width, tr_label, img_xt, label_xt, label_xt_swap,) = train_data_list
        tr_wid = tr_wid.to(gpu)
        tr_img = tr_img.to(gpu)
        tr_img_width = tr_img_width.to(gpu)
        tr_label = tr_label.to(gpu)
        img_xt = img_xt.to(gpu)
        label_xt = label_xt.to(gpu)
        label_xt_swap = label_xt_swap.to(gpu)
        batch_size = tr_domain.shape[0]

        if mode == "rec_update":
            tr_img_rec = tr_img[:, 0:1, :, :]  # 8,50,64,200 choose one channel 8,1,64,200
            tr_img_rec = tr_img_rec.requires_grad_()
            tr_label_rec = tr_label[:, 0, :]  # 8,50,10 choose one channel 8,10
            pred_xt_tr = self.rec(
                tr_img_rec, tr_label_rec, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)),
            )

            tr_label_rec2 = tr_label_rec[:, 1:]  # remove <GO>
            l_rec_tr = crit(log_softmax(pred_xt_tr.reshape(-1, vocab_size)), tr_label_rec2.reshape(-1),)
            cer_func.add(pred_xt_tr, tr_label_rec2)
            l_rec_tr.backward()
            return l_rec_tr

        elif mode == "cla_update":
            tr_img_rec = tr_img[:, 0:1, :, :]  # 8,50,64,200 choose one channel 8,1,64,200
            tr_img_rec = tr_img_rec.requires_grad_()
            l_cla_tr = self.cla(tr_img_rec, tr_wid)
            l_cla_tr.backward()
            return l_cla_tr

        elif mode == "gen_update":
            self.iter_num += 1
            """dis loss"""
            f_xs = self.gen.enc_image(tr_img)  # b,512,8,27
            f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)  # b,4096  b,512,8,27
            f_mix = self.gen.mix(f_xs, f_embed)

            xg = self.gen.decode(f_mix, f_xt)  # translation b,1,64,128
            l_dis_ori = self.dis.calc_gen_loss(xg, text=label_xt, writer=tr_wid)

            # '''poco modi -> swap char'''
            f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
            f_mix_swap = self.gen.mix(f_xs, f_embed_swap)
            xg_swap = self.gen.decode(f_mix_swap, f_xt_swap)  # translation b,1,64,128
            l_dis_swap = self.dis.calc_gen_loss(xg_swap, text=label_xt_swap, writer=tr_wid)

            """total loss for patch loss"""
            # add dummies for logging-purposes
            full_l_dis_swap, full_l_dis_orig = (
                torch.zeros_like(l_dis_swap),
                torch.zeros_like(l_dis_ori),
            )
            if self.USE_FULL_GAN:
                full_l_dis_swap = self.dis_full.calc_gen_loss(xg_swap)
                full_l_dis_orig = self.dis_full.calc_gen_loss(xg)
                l_dis = torch.stack((l_dis_ori, l_dis_swap, full_l_dis_swap, full_l_dis_orig)).mean()
            else:
                l_dis = torch.stack((l_dis_ori, l_dis_swap)).mean()
            total_l_dis = torch.stack((l_dis_ori, l_dis_swap)).mean()
            total_l_dis_full = torch.stack((full_l_dis_swap, full_l_dis_orig)).mean()
            """writer classifier loss"""
            l_cla_ori = self.cla(xg, tr_wid)
            l_cla_swap = self.cla(xg_swap, tr_wid)
            l_cla = (l_cla_ori + l_cla_swap) / 2.0

            """l1 loss"""
            if self.oov:
                l_l1 = torch.tensor(0.0).to(gpu)
            else:
                l_l1 = recon_criterion(xg, img_xt)

            """rec loss"""
            cer_te, cer_te2 = cer_func
            pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)),)
            pred_xt_swap = self.rec(
                xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)),
            )

            label_xt2 = label_xt[:, 1:]  # remove <GO>
            label_xt2_swap = label_xt_swap[:, 1:]  # remove <GO>
            l_rec_ori = crit(log_softmax(pred_xt.reshape(-1, vocab_size)), label_xt2.reshape(-1))
            l_rec_swap = crit(log_softmax(pred_xt_swap.reshape(-1, vocab_size)), label_xt2_swap.reshape(-1),)
            cer_te.add(pred_xt, label_xt2)
            cer_te2.add(pred_xt_swap, label_xt2_swap)
            l_rec = (l_rec_ori + l_rec_swap) / 2.0
            """fin"""
            l_total = w_dis * l_dis + w_cla * l_cla + w_l1 * l_l1 + w_rec * l_rec
            l_total.backward()
            return l_total, total_l_dis, total_l_dis_full, l_cla, l_l1, l_rec

        elif mode == "dis_update":
            sample_img1 = tr_img[:, 0:1, :, :]
            sample_img2 = tr_img[:, 1:2, :, :]
            # print("sample images", sample_img1.shape, sample_img2.shape)
            # print("sample image total", tr_img.shape)
            sample_img1.requires_grad_()
            sample_img2.requires_grad_()
            # similar to the sample images above, we only use the relevant string for recognizing
            l_real1 = self.dis.calc_dis_real_loss(sample_img1, text=tr_label[:, 0:1, :].squeeze(1), writer=tr_wid)
            l_real2 = self.dis.calc_dis_real_loss(sample_img2, text=tr_label[:, 1:2, :].squeeze(1), writer=tr_wid)

            l_real = torch.stack((l_real1, l_real2)).mean()
            l_real.backward(retain_graph=True)

            with torch.no_grad():
                f_xs = self.gen.enc_image(tr_img)
                f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)
                f_mix = self.gen.mix(f_xs, f_embed)
                xg = self.gen.decode(f_mix, f_xt)
                # swap tambien
                f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
                f_mix_swap = self.gen.mix(f_xs, f_embed_swap)
                xg_swap = self.gen.decode(f_mix_swap, f_xt_swap)

            l_fake_ori = self.dis.calc_dis_fake_loss(xg, text=label_xt, writer=tr_wid)
            l_fake_swap = self.dis.calc_dis_fake_loss(xg_swap, text=label_xt_swap, writer=tr_wid)

            l_fake = torch.stack((l_fake_ori, l_fake_swap)).mean()
            l_fake.backward()
            full_l_fake, full_l_real = torch.tensor([0]), torch.tensor([0])
            if self.USE_FULL_GAN:
                full_l_real1 = self.dis_full.calc_dis_real_loss(sample_img1)
                full_l_real2 = self.dis_full.calc_dis_real_loss(sample_img1)
                full_l_real = torch.stack((full_l_real1, full_l_real2)).mean()
                full_l_real.backward(retain_graph=True)
                full_l_fake_orig = self.dis_full.calc_dis_fake_loss(xg)
                full_l_fake_swap = self.dis_full.calc_dis_fake_loss(xg_swap)
                full_l_fake = torch.stack((full_l_fake_orig, full_l_fake_swap)).mean()
                full_l_fake.backward()

            l_total = l_real + l_fake
            l_full_total = full_l_fake + full_l_real
            """write images"""
            if self.iter_num % self.show_iter_num == 0:
                with torch.no_grad():
                    pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)),)
                    pred_xt_swap = self.rec(
                        xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)),
                    )
                write_image(
                    xg,
                    pred_xt,
                    img_xt,
                    label_xt,
                    tr_img,
                    xg_swap,
                    pred_xt_swap,
                    label_xt_swap,
                    "epoch_" + str(epoch) + "-" + str(self.iter_num),
                )
            return l_total, l_full_total

        elif mode == "eval":
            with torch.no_grad():
                f_xs = self.gen.enc_image(tr_img)
                f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)
                f_mix = self.gen.mix(f_xs, f_embed)
                xg = self.gen.decode(f_mix, f_xt)
                # second oov word
                f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
                f_mix_swap = self.gen.mix(f_xs, f_embed_swap)
                xg_swap = self.gen.decode(f_mix_swap, f_xt_swap)
                """write images"""
                pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)),)
                pred_xt_swap = self.rec(
                    xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)),
                )
                write_image(
                    xg,
                    pred_xt,
                    img_xt,
                    label_xt,
                    tr_img,
                    xg_swap,
                    pred_xt_swap,
                    label_xt_swap,
                    "eval_" + str(epoch) + "-" + str(self.iter_num),
                )
                self.iter_num += 1
                """dis loss"""
                l_dis_ori = self.dis.calc_gen_loss(xg, text=label_xt, writer=tr_wid)
                l_dis_swap = self.dis.calc_gen_loss(xg_swap, text=label_xt_swap, writer=tr_wid)

                l_dis = (l_dis_ori + l_dis_swap) / 2.0
                l_dis_full = torch.tensor([0])
                if self.USE_FULL_GAN:
                    l_dis_ori_full = self.dis_full.calc_dis_real_loss(xg)
                    l_dis_swap_full = self.dis_full.calc_dis_real_loss(xg_swap)
                    l_dis_full = (l_dis_ori_full + l_dis_swap_full) / 2.0
                """rec loss"""
                cer_te, cer_te2 = cer_func
                label_xt2 = label_xt[:, 1:]  # remove <GO>
                label_xt2_swap = label_xt_swap[:, 1:]  # remove <GO>
                l_rec_ori = crit(log_softmax(pred_xt.reshape(-1, vocab_size)), label_xt2.reshape(-1))
                l_rec_swap = crit(log_softmax(pred_xt_swap.reshape(-1, vocab_size)), label_xt2_swap.reshape(-1),)
                cer_te.add(pred_xt, label_xt2)
                cer_te2.add(pred_xt_swap, label_xt2_swap)
                l_rec = (l_rec_ori + l_rec_swap) / 2.0

                """writer classifier loss"""
                l_cla_ori = self.cla(xg, tr_wid)
                l_cla_swap = self.cla(xg_swap, tr_wid)
                l_cla = (l_cla_ori + l_cla_swap) / 2.0

            return l_dis, l_dis_full, l_cla, l_rec
