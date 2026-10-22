import torch
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
import torch.utils.benchmark as benchmark
from MixDehazeNet import MixDehazeNet_s
# 确保这里的导入正确
from MixDehazeNet import MixDehazeNet_l
# 创建模型实例（这里以 MixDehazeNet_s 为例）
model = MixDehazeNet_s()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 计算参数量
# 设置输入尺寸（例如输入为 4 通道的 256x256 图像）
input_tensor = torch.randn(1, 4, 256, 256).to(device)  # 确保输入也在正确的设备上

# 调用 summary 查看模型概况
print("Model Summary:")
summary(model, input_size=(1, 4, 256, 256))

# 计算 MACs
flops = FlopCountAnalysis(model, input_tensor)
print(f"MACs (in GFLOPS): {flops.total() / 1e9:.4f} GFLOPS")

# 计算延迟
timer = benchmark.Timer(
    stmt="model(input_tensor)",  # 被测试的代码
    setup="import torch",  # 不需要重新定义 model，globals 中已传递
    globals={"model": model, "input_tensor": input_tensor},  # 将 model 和 input_tensor 正确传递
    num_threads=1  # 可以根据硬件环境调整线程数
)

# 测量延迟：执行 100 次前向传播，计算平均时间
latency = timer.timeit(100)  # 测量 100 次前向传播的时间
print(f"Latency (ms): {latency.mean * 1e3:.4f} ms")  # 将秒转为毫秒
