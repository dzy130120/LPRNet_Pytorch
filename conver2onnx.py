# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''
from data.load_data import CHARS
from model.LPRNet import LPRNet
import torch


def conver2onnx():

    lprnet = LPRNet(8, False, len(CHARS), 0)
    device = torch.device("cuda:0")
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('./weights/Final_LPRNet_model.pth'))
    dummy_input = torch.randn(1, 3, 24, 94, device='cuda')
    torch.onnx.export(lprnet, dummy_input, "LPR.onnx", verbose=True)
    print("Successful to conver2onnx")

if __name__ == "__main__":
    conver2onnx()
