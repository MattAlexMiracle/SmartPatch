import numpy as np
import os
import torch
from torch import nn
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from vgg_tro_channel3_modi import vgg19_bn
from recognizer.models.encoder_vgg import Encoder as rec_encoder
from recognizer.models.decoder import Decoder as rec_decoder
from recognizer.models.seq2seq import Seq2Seq as rec_seq2seq
from recognizer.models.attention import locationAttention as rec_attention
from load_data import (
    OUTPUT_MAX_LEN,
    IMG_HEIGHT,
    IMG_WIDTH,
    DISPLAY_IMG_HEIGHT,
    DISPLAY_IMG_WIDTH,
    vocab_size,
    index2letter,
    num_tokens,
    tokens,
    NUM_WRITERS,
)
import cv2
import torch.nn.functional as F
import torchvision
from load_data import NUM_CHANNEL

try:
    import kornia
    from PIL import Image
    import time
except ImportError:
    pass

gpu = torch.device("cuda")
PATCH_WIDTH = 64  # 32 for reduced resolution
PATCH_HEIGHT = IMG_HEIGHT
WINDOW_STEPSIZE_HEIGHT = PATCH_HEIGHT
WINDOW_STEPSIZE_WIDTH = 16
image_folder = None


def split_into_patches(imgs, patch_width, patch_height, use_full_image=False):
    imgs = F.pad(imgs, (WINDOW_STEPSIZE_WIDTH, WINDOW_STEPSIZE_WIDTH))
    vertical = imgs.unfold(-2, patch_height, WINDOW_STEPSIZE_HEIGHT)
    horizontal = vertical.unfold(-2, patch_width, WINDOW_STEPSIZE_WIDTH)
    batch, dim, a, b, height, width = horizontal.shape
    horizontal = horizontal.reshape(batch, dim, a * b, height, width)
    horizontal = horizontal.permute(0, 2, 1, 3, 4)
    horizontal = horizontal.flatten(end_dim=1)
    # Image.fromarray(kornia.tensor_to_image(horizontal[0]) * 255).show()
    # print(horizontal.shape)
    # if use_full_image:
    #    chunks.append(torch.clone(imgs))
    return horizontal


def normalize(tar):
    tar = (tar - tar.min()) / (tar.max() - tar.min())
    tar = tar * 255
    tar = tar.astype(np.uint8)
    return tar


def fine(label_list):
    if type(label_list) != type([]):
        return [label_list]
    else:
        return label_list


def write_image(
    xg, pred_label, gt_img, gt_label, tr_imgs, xg_swap, pred_label_swap, gt_label_swap, title, num_tr=2,
):
    folder = image_folder
    gt_img = F.interpolate(gt_img, size=(DISPLAY_IMG_HEIGHT, DISPLAY_IMG_WIDTH))
    tr_imgs = F.interpolate(tr_imgs, size=(DISPLAY_IMG_HEIGHT, DISPLAY_IMG_WIDTH))
    xg = F.interpolate(xg, size=(DISPLAY_IMG_HEIGHT, DISPLAY_IMG_WIDTH))
    xg_swap = F.interpolate(xg_swap, size=(DISPLAY_IMG_HEIGHT, DISPLAY_IMG_WIDTH))
    if not os.path.exists(folder):
        os.makedirs(folder)
    batch_size = gt_label.shape[0]
    tr_imgs = tr_imgs.cpu().numpy()
    xg = xg.cpu().numpy()
    xg_swap = xg_swap.cpu().numpy()
    gt_img = gt_img.cpu().numpy()
    gt_label = gt_label.cpu().numpy()
    gt_label_swap = gt_label_swap.cpu().numpy()
    pred_label = torch.topk(pred_label, 1, dim=-1)[1].squeeze(-1)  # b,t,83 -> b,t,1 -> b,t
    pred_label = pred_label.cpu().numpy()
    pred_label_swap = torch.topk(pred_label_swap, 1, dim=-1)[1].squeeze(-1)  # b,t,83 -> b,t,1 -> b,t
    pred_label_swap = pred_label_swap.cpu().numpy()
    tr_imgs = tr_imgs[:, :num_tr, :, :]
    outs = list()
    for i in range(batch_size):
        src = tr_imgs[i].reshape(num_tr * DISPLAY_IMG_HEIGHT, -1)
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        tar_swap = xg_swap[i].squeeze()
        src = normalize(src)
        gt = normalize(gt)
        tar = normalize(tar)
        tar_swap = normalize(tar_swap)
        gt_text = gt_label[i].tolist()
        gt_text_swap = gt_label_swap[i].tolist()
        pred_text = pred_label[i].tolist()
        pred_text_swap = pred_label_swap[i].tolist()

        gt_text = fine(gt_text)
        gt_text_swap = fine(gt_text_swap)
        pred_text = fine(pred_text)
        pred_text_swap = fine(pred_text_swap)

        for j in range(num_tokens):
            gt_text = list(filter(lambda x: x != j, gt_text))
            gt_text_swap = list(filter(lambda x: x != j, gt_text_swap))
            pred_text = list(filter(lambda x: x != j, pred_text))
            pred_text_swap = list(filter(lambda x: x != j, pred_text_swap))

        gt_text = "".join([index2letter[c - num_tokens] for c in gt_text])
        gt_text_swap = "".join([index2letter[c - num_tokens] for c in gt_text_swap])
        pred_text = "".join([index2letter[c - num_tokens] for c in pred_text])
        pred_text_swap = "".join([index2letter[c - num_tokens] for c in pred_text_swap])
        gt_text_img = np.zeros_like(tar)
        gt_text_img_swap = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        pred_text_img_swap = np.zeros_like(tar)
        cv2.putText(
            gt_text_img, gt_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        cv2.putText(
            gt_text_img_swap, gt_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        cv2.putText(
            pred_text_img, pred_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        cv2.putText(
            pred_text_img_swap, pred_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        out = np.vstack([src, gt, gt_text_img, tar, pred_text_img, gt_text_img_swap, tar_swap, pred_text_img_swap,])
        outs.append(out)
    final_out = np.hstack(outs)
    cv2.imwrite(folder + "/" + title + ".png", final_out)


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


class DisModelPatch(nn.Module):
    def __init__(self, input_nc=1, n_layers=2, additional_cap=False):
        """
        Architecture similar to
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        """
        super(DisModelPatch, self).__init__()
        ndf = 64
        norm_layer = nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        use_bias = False
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if additional_cap:
                sequence += [
                    # ActFirstResBlock(
                    #    ndf * nf_mult_prev, ndf * nf_mult_prev, None, "lrelu", "none"
                    # ),
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult_prev, 3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(
                        ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    # ActFirstResBlock(
                    #    ndf * nf_mult_prev, ndf * nf_mult_prev, None, "lrelu", "none"
                    # ),
                    # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult_prev, 3, padding=1),
                    # nn.LeakyReLU(0.2, True),
                    nn.Conv2d(
                        ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            # ActFirstResBlock(
            #    ndf * nf_mult_prev, ndf * nf_mult_prev, None, "lrelu", "none"
            # ),
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias,),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.model(x)
        # print(out.shape)
        return out.squeeze(-1).squeeze(-1)  # b,1024   maybe b is also 1, so cannnot out.squeeze()

    def discriminator_patch_loss(self, images, label, **kwargs):
        # patch-loss
        patches = split_into_patches(images, PATCH_WIDTH, PATCH_HEIGHT)
        # patches = images
        out_disc = self.forward(patches)
        label = torch.ones_like(out_disc) * label
        loss = self.bce(out_disc, label)

        return loss

    def calc_dis_fake_loss(self, input_fake, **kwargs):
        label = 0
        return self.discriminator_patch_loss(input_fake, label, **kwargs)

    def calc_dis_real_loss(self, input_real, **kwargs):
        label = 1
        return self.discriminator_patch_loss(input_real, label, **kwargs)

    def calc_gen_loss(self, input_fake, **kwargs):
        label = 1
        return self.discriminator_patch_loss(input_fake, label, **kwargs)


class DisModel(nn.Module):
    def __init__(self):
        """
        Traditional GANwriting Discriminator Model
        """
        super(DisModel, self).__init__()
        self.n_layers = 6  # 5 reduce when imagesize is reduced, for original use 6
        self.final_size = 1024
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3, pad_type="reflect", norm="none", activation="none")]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, "lrelu", "none")]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, "lrelu", "none")]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, "lrelu", "none")]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, "lrelu", "none")]
        cnn_c = [
            Conv2dBlock(
                nf_out,
                self.final_size,
                IMG_HEIGHT // (2 ** (self.n_layers - 1)),
                IMG_WIDTH // (2 ** (self.n_layers - 1)) + 1,
                norm="none",
                activation="lrelu",
                activation_first=True,
            )
        ]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        return out.squeeze(-1).squeeze(-1)  # b,1024   maybe b is also 1, so cannnot out.squeeze()

    def calc_dis_fake_loss(self, input_fake, **kwargs):
        label = torch.zeros(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def calc_dis_real_loss(self, input_real, **kwargs):
        label = torch.ones(input_real.shape[0], self.final_size).to(gpu)
        resp_real = self.forward(input_real)
        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake, **kwargs):
        label = torch.ones(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss


class DisModelPatchSmart(DisModelPatch):
    def __init__(self, recognition_model, input_nc=1, n_layers=2, additional_cap=False):
        """
        Architecture similar to
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        """
        super(DisModelPatchSmart, self).__init__(input_nc=input_nc, n_layers=n_layers, additional_cap=additional_cap)
        self.recognition_model = recognition_model
        self.iters = 0

    def _get_patch_from_encoder(self, images):
        with torch.no_grad():
            multi_image = torch.cat([images, images, images], dim=1)
            # the label doesn't matter, since it's only used for seeding
            out, attn_weights = self.recognition_model.seq2seq(
                multi_image,
                torch.tensor([tokens["GO_TOKEN"]] * len(images)).unsqueeze(1).to(gpu),
                src_len=torch.from_numpy(np.array([IMG_WIDTH] * len(images))),
                teacher_rate=False,
                train=False,
            )
        # upsample attn_weights to IMG_WIDTH
        # upsampling only implemented for 3D and up, add artificial "Channel",
        # use align_corners=False because of empirically better results
        attn_weights = torch.cat(
            [F.interpolate(x.unsqueeze(1), IMG_WIDTH, mode="linear", align_corners=False) for x in attn_weights], 1,
        )
        assert attn_weights.shape[0] == len(images)
        assert attn_weights.shape[2] == IMG_WIDTH
        indices = attn_weights.argmax(2)
        # pad for enough space left and right
        padded_img = F.pad(images, (PATCH_WIDTH, PATCH_WIDTH))
        batchsize, timesteps, _ = attn_weights.shape
        patches = []
        out = out.permute(1, 0, 2)
        for i in range(batchsize):
            for j in range(timesteps):

                # select an equal amount of space left and right
                # (assumes higest window-value in the middle of the character)
                patches.append(
                    padded_img[
                        i,
                        :,
                        :,
                        indices[i, j]
                        + np.floor(PATCH_WIDTH / 2).astype(int) : indices[i, j]
                        + np.ceil(1.5 * PATCH_WIDTH).astype(int),
                    ]
                )

                # if self.iters % 100 == 0:
                #    Image.fromarray(kornia.tensor_to_image(patches[-1]) * 255).resize(
                #        (128, 128)
                #    ).show()
                #    print(
                #        index2letter[torch.argmax(out[i, j]).cpu().item() - num_tokens]
                #    )
                #    self.iters += 1

                if torch.argmax(out[i, j]) == tokens["END_TOKEN"]:
                    # break once "end token" appears, at least one item is needed
                    break

        return torch.stack(patches)

    def discriminator_patch_loss(self, images, label, **kwargs):
        # raise NotImplementedError
        # patch-loss
        self.iters += 1
        patches = self._get_patch_from_encoder(images)
        # if self.iters % 100 == 0 and label == 1 and self.iters > 5_000:
        #    Image.fromarray(kornia.tensor_to_image(patches[0]) * 255).show()

        # Image.fromarray(kornia.tensor_to_image(patches[0]) * 255).show()
        out_disc = self.forward(patches)
        label = torch.ones_like(out_disc) * label
        loss = self.bce(out_disc, label)
        # losses = torch.chunk(loss, nr_patches, 0)
        # total_loss = torch.stack(losses).mean(0)
        return loss


class DisModelPatchWithCharacters(DisModelPatch):
    def __init__(
        self, recognition_model, input_nc=1, n_layers=2, embedding_dim=8, additional_cap=False,
    ):
        """
        Architecture similar to
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        """
        super(DisModelPatchWithCharacters, self).__init__(input_nc=input_nc, n_layers=n_layers)
        ndf = 64
        norm_layer = nn.InstanceNorm2d
        kw = 4
        padw = 1
        self.n_layers = n_layers
        self.recognition_model = recognition_model
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(vocab_size, embedding_dim)
        self.conv_down = nn.Conv2d(embedding_dim + input_nc, input_nc, 1)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        use_bias = False
        nf_mult = 1
        nf_mult_prev = 1
        self.injection_point = 0
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            embedd = self.embedding_dim if n == n_layers - 1 else 0
            self.injection_point = len(sequence) if n == n_layers - 1 else self.injection_point
            channels = ndf * nf_mult_prev + embedd
            if additional_cap:
                sequence += [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(channels, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias,),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(channels, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias,),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias,),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.ModuleList(sequence)
        self.iters = 0
        self.bce = nn.BCEWithLogitsLoss()

    def _get_patch_from_encoder(self, images, text=None):

        with torch.no_grad():
            multi_image = torch.cat([images, images, images], dim=1)
            # the label doesn't matter, since it's only used for seeding
            out, attn_weights = self.recognition_model.seq2seq(
                multi_image,
                torch.tensor([tokens["GO_TOKEN"]] * len(images)).unsqueeze(1).to(gpu),
                src_len=torch.from_numpy(np.array([IMG_WIDTH] * len(images))),
                teacher_rate=False,
                train=False,
            )
        # upsample attn_weights to IMG_WIDTH
        # upsampling only implemented for 3D and up, add artificial "Channel"
        attn_weights = torch.cat(
            [F.interpolate(x.unsqueeze(1), IMG_WIDTH, mode="linear", align_corners=False) for x in attn_weights], 1,
        )
        assert attn_weights.shape[0] == len(images)
        assert attn_weights.shape[2] == IMG_WIDTH
        indices = attn_weights.argmax(2)
        # pad for enough space left and right
        padded_img = F.pad(images, (PATCH_WIDTH, PATCH_WIDTH))
        batchsize, timesteps, _ = attn_weights.shape
        patches, characters = [], []
        out = out.permute(1, 0, 2)
        for i in range(batchsize):
            for j in range(timesteps):
                # select an equal amount of space left and right
                patches.append(
                    padded_img[
                        i,
                        :,
                        :,
                        indices[i, j].item()
                        + np.floor(PATCH_WIDTH / 2).astype(int) : indices[i, j].item()
                        + np.ceil(1.5 * PATCH_WIDTH).astype(int),
                    ]
                )
                # the first token is always the <GO_TOKEN>, so disregard that
                characters.append(text[i, j + 1])
                # Image.fromarray(kornia.tensor_to_image(patches[-1]) * 255).resize(
                #    (128, 128)
                # ).show()
                # print(
                #    i,
                #    j,
                #    index2letter[torch.argmax(out[i, j]).cpu().item() - num_tokens],
                # )
                # print(index2letter[text[i, j + 1].cpu().item() - num_tokens])
                # time.sleep(1)
                # use the original text for character info, but the recognized text
                # for limiting the number of windows created. In case the original word
                # has fewer letters than the predicted one, also end (happens rarely after the first ~5 epochs)
                if torch.argmax(out[i, j]) == tokens["END_TOKEN"] or text[i, j + 2] == tokens["END_TOKEN"]:
                    # break once "end token" appears, at least one item is needed
                    break
        characters = torch.stack(characters)
        one_hot = torch.zeros(len(characters), vocab_size).to(gpu)
        one_hot.scatter_(1, characters.unsqueeze(1), 1)
        return torch.stack(patches), one_hot

    def discriminator_patch_loss(self, images, label, **kwargs):
        # patch-loss

        # self.iters += 1
        patches, characters = self._get_patch_from_encoder(images, text=kwargs.get("text"))
        # if self.iters % 100 == 0 and label == 1 and self.iters > 250_000:
        #    Image.fromarray(kornia.tensor_to_image(patches[0]) * 255).show()
        embedded = F.leaky_relu(self.embedder(characters))

        for i in range(len(self.model)):
            if i == self.injection_point:
                resized_embedd = (
                    embedded.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(len(embedded), self.embedding_dim, patches.shape[-2], patches.shape[-1],)
                )
                patches = torch.cat((resized_embedd, patches), dim=1)
                patches = self.model[i](patches)
            else:
                patches = self.model[i](patches)
            # print(i, len(self.model), patches.shape)

        label = torch.ones_like(patches) * label
        loss = self.bce(patches, label)
        # losses = torch.chunk(loss, nr_patches, 0)
        # total_loss = torch.stack(losses).mean(0)
        return loss


class DisModelPatchWithStyle(DisModelPatch):
    def __init__(
        self, recognition_model, input_nc=1, n_layers=2, embedding_dim=8, additional_cap=False,
    ):
        """
        Architecture similar to
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        """
        super(DisModelPatchWithStyle, self).__init__(input_nc=input_nc, n_layers=n_layers)
        ndf = 64
        norm_layer = nn.InstanceNorm2d
        kw = 4
        padw = 1
        self.n_layers = n_layers
        self.recognition_model = recognition_model
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(NUM_WRITERS, embedding_dim)
        self.conv_down = nn.Conv2d(embedding_dim + input_nc, input_nc, 1)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        use_bias = False
        nf_mult = 1
        nf_mult_prev = 1
        self.injection_point = 0
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            embedd = self.embedding_dim if n == n_layers - 1 else 0
            self.injection_point = len(sequence) if n == n_layers - 1 else self.injection_point
            channels = ndf * nf_mult_prev + embedd
            if additional_cap:
                sequence += [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(channels, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias,),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(channels, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias,),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias,),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.ModuleList(sequence)
        self.iters = 0
        self.bce = nn.BCEWithLogitsLoss()

    def _get_patch_from_encoder(self, images, writer=None):

        with torch.no_grad():
            multi_image = torch.cat([images, images, images], dim=1)
            # the label doesn't matter, since it's only used for seeding
            out, attn_weights = self.recognition_model.seq2seq(
                multi_image,
                torch.tensor([tokens["GO_TOKEN"]] * len(images)).unsqueeze(1).to(gpu),
                src_len=torch.from_numpy(np.array([IMG_WIDTH] * len(images))),
                teacher_rate=False,
                train=False,
            )
        # upsample attn_weights to IMG_WIDTH
        # upsampling only implemented for 3D and up, add artificial "Channel"
        attn_weights = torch.cat(
            [F.interpolate(x.unsqueeze(1), IMG_WIDTH, mode="linear", align_corners=False) for x in attn_weights], 1,
        )
        assert attn_weights.shape[0] == len(images)
        assert attn_weights.shape[2] == IMG_WIDTH
        indices = attn_weights.argmax(2)
        # pad for enough space left and right
        padded_img = F.pad(images, (PATCH_WIDTH, PATCH_WIDTH))
        batchsize, timesteps, _ = attn_weights.shape
        patches, writer_ids = [], []
        out = out.permute(1, 0, 2)
        for i in range(batchsize):
            for j in range(timesteps):
                # select an equal amount of space left and right
                patches.append(
                    padded_img[
                        i,
                        :,
                        :,
                        indices[i, j]
                        + np.floor(PATCH_WIDTH / 2).astype(int) : indices[i, j]
                        + np.ceil(1.5 * PATCH_WIDTH).astype(int),
                    ]
                )
                writer_ids.append(writer[i])
                # use the original text for character info, but the recognized text
                # for limiting the number of windows created
                if torch.argmax(out[i, j]) == tokens["END_TOKEN"]:
                    # break once "end token" appears, at least one item is needed
                    break
        writer_ids = torch.stack(writer_ids)
        one_hot = torch.zeros(len(writer_ids), NUM_WRITERS).to(gpu)
        one_hot.scatter_(1, writer_ids.unsqueeze(1), 1)
        # print("one-hot", one_hot.shape)
        return torch.stack(patches), one_hot

    def discriminator_patch_loss(self, images, label, **kwargs):
        # patch-loss

        # self.iters += 1
        patches, writer = self._get_patch_from_encoder(images, writer=kwargs.get("writer"))
        # if self.iters % 100 == 0 and label == 1 and self.iters > 250_000:
        #    Image.fromarray(kornia.tensor_to_image(patches[0]) * 255).show()
        embedded = F.leaky_relu(self.embedder(writer))

        for i in range(len(self.model)):
            if i == self.injection_point:
                resized_embedd = (
                    embedded.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(len(embedded), self.embedding_dim, patches.shape[-2], patches.shape[-1],)
                )
                patches = torch.cat((resized_embedd, patches), dim=1)
                patches = self.model[i](patches)
                # 256+4 channels when using 3 layers
            else:
                patches = self.model[i](patches)
            # print(i, len(self.model), patches.shape)

        label = torch.ones_like(patches) * label
        loss = self.bce(patches, label)
        # losses = torch.chunk(loss, nr_patches, 0)
        # total_loss = torch.stack(losses).mean(0)
        return loss


class WriterClaModel(nn.Module):
    def __init__(self, num_writers):
        super(WriterClaModel, self).__init__()
        self.n_layers = 6  # 5 for testing in lower res, originally res is 6
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3, pad_type="reflect", norm="none", activation="none")]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, "lrelu", "none")]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, "lrelu", "none")]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, "lrelu", "none")]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, "lrelu", "none")]
        cnn_c = [
            Conv2dBlock(
                nf_out,
                num_writers,
                IMG_HEIGHT // (2 ** (self.n_layers - 1)),
                IMG_WIDTH // (2 ** (self.n_layers - 1)) + 1,
                norm="none",
                activation="lrelu",
                activation_first=True,
            )
        ]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)  # b,310,1,1
        loss = self.cross_entropy(out.squeeze(-1).squeeze(-1), y)
        return loss


class GenModel_FC(nn.Module):
    def __init__(self, text_max_len):
        super(GenModel_FC, self).__init__()
        self.enc_image = ImageEncoder().to(gpu)
        self.enc_text = TextEncoder_FC(text_max_len).to(gpu)
        self.dec = Decoder().to(gpu)
        self.linear_mix = nn.Linear(1024, 512)

    def decode(self, content, adain_params):
        # decode content and style codes to an image
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    # feat_mix: b,1024,8,27
    def mix(self, feat_xs, feat_embed):
        feat_mix = torch.cat([feat_xs, feat_embed], dim=1)  # b,1024,8,27
        f = feat_mix.permute(0, 2, 3, 1)
        ff = self.linear_mix(f)  # b,8,27,1024->b,8,27,512
        return ff.permute(0, 3, 1, 2)


class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len):
        super(TextEncoder_FC, self).__init__()
        embed_size = 64
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
            nn.Linear(text_max_len * embed_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=False),
            nn.Linear(2048, 4096),
        )
        """embed content force"""
        self.linear = nn.Linear(embed_size, 512)

    def forward(self, x, f_xs_shape):
        xx = self.embed(x)  # b,t,embed
        batch_size = xx.shape[0]
        xxx = xx.reshape(batch_size, -1)  # b,t*embed
        out = self.fc(xxx)

        """embed content force"""
        xx_new = self.linear(xx)  # b, text_max_len, 512
        ts = xx_new.shape[1]
        height_reps = f_xs_shape[-2]
        width_reps = f_xs_shape[-1] // ts
        tensor_list = list()
        for i in range(ts):
            text = [xx_new[:, i : i + 1]]  # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = f_xs_shape[-1] % ts
        if padding_reps:
            embedded_padding_char = self.embed(torch.full((1, 1), tokens["PAD_TOKEN"], dtype=torch.long).cuda())
            embedded_padding_char = self.linear(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(tensor_list, dim=1)  # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(2)  # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)

        return out, final_res


"""VGG19_IN tro"""


def batchnorm_to_instanceNorm(model):
    for idx, x in enumerate(model):
        if isinstance(x, nn.BatchNorm2d):
            model[idx] = nn.InstanceNorm2d(x.num_features)


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False)

        self.output_dim = 512

    def forward(self, x):
        # print("encoding", x.shape)
        # print(self.model_old(x).shape)
        # print(self.model(x).shape)
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self, ups=3, n_res=2, dim=512, out_dim=1, res_norm="adain", activ="relu", pad_type="reflect",
    ):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm="in", activation=activ, pad_type=pad_type,),
            ]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3, norm="none", activation="tanh", pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class RecModel(nn.Module):
    def __init__(self, pretrain=False):
        super(RecModel, self).__init__()
        hidden_size_enc = hidden_size_dec = 512
        embed_size = 60
        self.enc = rec_encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(gpu)
        self.dec = rec_decoder(hidden_size_dec, embed_size, vocab_size, rec_attention, None).to(gpu)
        self.seq2seq = rec_seq2seq(self.enc, self.dec, OUTPUT_MAX_LEN, vocab_size).to(gpu)
        if pretrain:

            model_file = "recognizer/save_weights/seq2seq-72.model_5.79.bak"
            print("Loading RecModel", model_file)
            self.seq2seq.load_state_dict(torch.load(model_file))
            exit(-1)

    def forward(self, img, label, img_width):
        self.seq2seq.train()
        img = torch.cat([img, img, img], dim=1)  # b,1,64,128->b,3,64,128
        output, attn_weights = self.seq2seq(img, label, img_width, teacher_rate=False, train=False)
        return output.permute(1, 0, 2)  # t,b,83->b,t,83


class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3, norm="none", activ="relu"):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim, norm="none", activation="none")]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
