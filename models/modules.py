import torch

from torchvision import models


def make_decoder_block(in_channels, middle_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, middle_channels, 3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(
            middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True))
    return block


class UNet(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        encoder = models.vgg16_bn(pretrained=pretrained).features
        self.conv1 = encoder[:6]
        self.conv2 = encoder[6:13]
        self.conv3 = encoder[13:23]
        self.conv4 = encoder[23:33]
        self.conv5 = encoder[33:43]

        self.center = torch.nn.Sequential(
            encoder[43],  # MaxPool
            make_decoder_block(512, 512, 256)
        )

        self.dec5 = make_decoder_block(256 + 512, 512, 256)
        self.dec4 = make_decoder_block(256 + 512, 512, 256)
        self.dec3 = make_decoder_block(256 + 256, 256, 64)
        self.dec2 = make_decoder_block(64 + 128, 128, 32)
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(32 + 64, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.final = torch.nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(conv5)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(dec1)


class ResUNet(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        encoder = models.resnet34(pretrained=pretrained)
        self.conv1 = torch.nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        )
        self.conv2 = torch.nn.Sequential(
            encoder.maxpool,
            encoder.layer1
        )
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        self.center = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            make_decoder_block(512, 512, 512)
        )

        # self.center = torch.nn.Identity()

        self.dec5 = make_decoder_block(512 + 512, 512, 256)
        self.dec4 = make_decoder_block(256 + 256, 256, 128)
        self.dec3 = make_decoder_block(128 + 128, 128, 64)
        self.dec2 = make_decoder_block(64 + 64, 64, 64)
        self.dec1 = make_decoder_block(64 + 64, 64, 32)
        self.final = torch.nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(conv5)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class ResUNetLarge(ResUNet):
    def __init__(self, pretrained=False):
        super().__init__(pretrained=pretrained)
        self.dec5 = make_decoder_block(512 + 512, 512, 256)
        self.dec4 = make_decoder_block(256 + 256, 512, 128)
        self.dec3 = make_decoder_block(128 + 128, 256, 64)
        self.dec2 = make_decoder_block(64 + 64, 128, 64)
        self.dec1 = make_decoder_block(64 + 64, 128, 64)
        self.final = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(conv5)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)
