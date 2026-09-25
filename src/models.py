"""
Model architectures for fault segmentation.
Uses segmentation_models_pytorch (SMP) for UNet, UNet++, DeepLabV3+.
Also SegFormer via SMP.

Verified reference solution uses smp.Unet.
We extend to ensemble of architectures for leaderboard top.
"""

import torch
import torch.nn as nn


def get_model(arch="unetplusplus", encoder="efficientnet-b5", in_channels=19, classes=1, pretrained=True):
    """Build a segmentation model.

    `in_channels` defaults to 19 because that is what the competition feature stack
    actually has (measured on the downloaded raster: gems-geodawn-numerical-features.tif
    carries 19 float32 bands, each with a band_name tag -- see data/evidence/ and the
    Data page of the generated site).  It used to default to 10, a number taken from the
    problem page's prose; every caller in this repo passes `in_channels` explicitly, so
    the wrong default never reached a run, but it was a trap for anyone importing
    get_model directly.
    """
    # Keep SMP optional for metric, data and submission-format utilities.  Importing
    # `src.train` only to use its split helper should not require the heavyweight model
    # stack; model construction below still fails clearly if an actual training run
    # omitted the required dependency.
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation-models-pytorch is required to build a model; install "
            "requirements.txt (or use the lightweight metric/test dependencies)"
        ) from exc

    def _mk(maker, **kw):
        """Build with ImageNet weights, falling back to random init if the weights
        download fails (the sandbox cannot reach the release-asset host; runners can).
        A hard failure here would silently turn a `pretrained: true` config into zero
        checkpoints, so the fallback is a warning, not an exception."""
        if pretrained:
            try:
                return maker(encoder_weights="imagenet", **kw)
            except Exception as e:  # noqa: BLE001 - any network/integrity error
                print(f"WARNING: pretrained weights for encoder '{kw.get('encoder_name')}' "
                      f"failed to download ({type(e).__name__}: {e}); training from scratch")
        return maker(encoder_weights=None, **kw)

    arch = arch.lower()
    if arch == "unet":
        model = _mk(smp.Unet, encoder_name=encoder, in_channels=in_channels,
                    classes=classes, activation=None)
    elif arch == "unetplusplus":
        model = _mk(smp.UnetPlusPlus, encoder_name=encoder, in_channels=in_channels,
                    classes=classes, activation=None)
    elif arch == "deeplabv3plus":
        model = _mk(smp.DeepLabV3Plus, encoder_name=encoder, in_channels=in_channels,
                    classes=classes, activation=None)
    elif arch == "segformer":
        # SegFormer uses mit encoders
        # Map encoder name
        if "mit" not in encoder:
            encoder = "mit_b2"
        model = _mk(smp.Segformer, encoder_name=encoder, in_channels=in_channels,
                    classes=classes, activation=None)
    elif arch == "fpn":
        model = _mk(smp.FPN, encoder_name=encoder, in_channels=in_channels,
                    classes=classes, activation=None)
    else:
        raise ValueError(f"Unknown arch {arch}")

    return model

class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0/len(models)] * len(models)
        self.weights = weights

    def forward(self, x):
        # Average logits
        logits = 0
        for model, w in zip(self.models, self.weights):
            logits = logits + model(x) * w
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
