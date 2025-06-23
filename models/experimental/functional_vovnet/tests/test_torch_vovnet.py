# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import timm
import pytest
import torch

from loguru import logger

from models.experimental.functional_vovnet.reference.vovnet import VoVNet
from models.utility_functions import comp_pcc, comp_allclose_and_pcc

"""
def imagenet_sample_input():
    path = "models/sample_data/ILSVRC2012_val_00048736.JPEG"

    im = Image.open(path)
    im = im.resize((224, 224))
    return transforms.ToTensor()(im).unsqueeze(0)

def imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels
"""


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", 1.0),),
)
def test_vovnet_image_classification_inference(
    model_name, pcc, imagenet_sample_input, imagenet_label_dict, reset_seeds
):
    Timm_model = timm.create_model(model_name, pretrained=True)
    Timm_model.eval()

    state_dict = Timm_model.state_dict()
    print("Shape of onput :", imagenet_sample_input.shape)

    PT_model = VoVNet()
    res = PT_model.load_state_dict(state_dict)
    # image = imagenet_sample_input()

    with torch.no_grad():
        Timm_output = Timm_model(imagenet_sample_input)
        PT_output = PT_model(imagenet_sample_input)

    Timm_logits = Timm_output[0]
    PT_logits = PT_output[0]

    passing, info = comp_pcc(Timm_logits, PT_logits, pcc)
    _, info = comp_allclose_and_pcc(Timm_logits, PT_logits, pcc)
    logger.info(info)

    if passing:
        logger.info("VovNetForImageClassification Passed!")
    else:
        logger.warning("VovNetForImageClassification Failed!")

    assert passing
