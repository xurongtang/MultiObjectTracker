import torch
from torchreid import models
import argparse

def main(weight_path, onnx_path, input_size=(256, 128)):
    # ✅ 正确：num_classes=751（Market1501）
    model = models.build_model(
        name='osnet_x1_0',
        num_classes=751,      # 👈 必须匹配训练时的类别数
        pretrained=False
    )
    model.eval()

    # 加载权重
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)  # 建议 strict=True 验证

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # 验证输出
    with torch.no_grad():
        feat = model(dummy_input)
        print("✅ PyTorch output shape:", feat.shape)  # 应为 [1, 512]

    # ✅ 直接导出 model，不要包装！
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ ONNX 模型已保存至: {onnx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--onnx', type=str, default='osnet_x1_0_market.onnx')
    args = parser.parse_args()
    main(args.weight, args.onnx)