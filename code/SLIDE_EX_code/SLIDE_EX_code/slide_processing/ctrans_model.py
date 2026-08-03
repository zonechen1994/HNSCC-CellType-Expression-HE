"""
CTransPath feature extractor for SLIDE-EX.

The paper's utils_preprocessing.py does `from ctrans_model import CTransPath`
and calls `CTransPath(num_classes=0)`. CTransPath is the TransPath Swin-Tiny
with a ConvStem patch embed, loaded from ctranspath.pth.

Requires the MODIFIED timm 0.5.4 (supports `embed_layer=` for swin). We vendor it
under models/vendored/timm-0.5.4 and prepend it to sys.path so it overrides any
system timm, without touching the rest of the environment.
"""
import os, sys

# --- make the vendored modified timm 0.5.4 win over any installed timm ---
_VENDORED_TIMM = os.environ.get(
    "CTRANS_TIMM_PATH",
    os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/models/vendored/timm-0.5.4",
)
if "timm" in sys.modules:
    raise RuntimeError("timm already imported before ctrans_model; import this first")
sys.path.insert(0, _VENDORED_TIMM)

import timm
import torch.nn as nn
from timm.models.layers.helpers import to_2tuple

assert timm.__version__ == "0.5.4", f"need modified timm 0.5.4, got {timm.__version__}"


class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True):
        super().__init__()
        assert patch_size == 4 and embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def CTransPath(num_classes=0):
    """Build the Swin-Tiny + ConvStem CTransPath model (no classification head)."""
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=ConvStem,
        pretrained=False,
        num_classes=num_classes,
    )
    return model
