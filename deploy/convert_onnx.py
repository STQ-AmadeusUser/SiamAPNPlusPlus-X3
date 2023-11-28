import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
from torch import nn
from pysot.models.model_builder_adapn_deploy import ModelBuilderADAPN
from pysot.utils.model_load import load_pretrain
import torch.onnx

batch_size = 1


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Generate SiamAPN++ ONNX')
    parser.add_argument('--cfg', type=str, default='../experiments/config.yaml',
                        help='configuration of tracking')
    parser.add_argument('--snapshot', default='../snapshot/SiamAPN++.pth', type=str,
                        help='snapshot of models to eval')
    args = parser.parse_args()

    return args


# reference: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
if __name__ == '__main__':

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build siamese network for tracking
    siam_onnx = ModelBuilderADAPN()
    siam_onnx = load_pretrain(siam_onnx, args.snapshot, device)
    siam_onnx.eval()

    # Input to the model
    torch_1 = torch.randn(batch_size, 3, 127, 127, requires_grad=True)
    torch_2 = torch.randn(batch_size, 3, 287, 287, requires_grad=True)
    torch_3, torch_4, torch_5, torch_6, torch_7 = siam_onnx(torch_1, torch_2)

    # Export the model
    torch.onnx.export(siam_onnx,  # model being run
                      (torch_1, torch_2),  # model input (or a tuple for multiple inputs)
                      "SiamAPN++.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input1', 'input2'],  # the model's input names
                      output_names=['output1', 'output2', 'output3', 'output4', 'output5'],  # the model's output names
                      )
