import torch
import platform
import subprocess

def check_intel_xpu_support():
    print("="*50)
    print("Intel XPU 支持检测")
    print("="*50)
    
    # 1. 检查操作系统
    system = platform.system()
    print(f"操作系统: {system}")
    
    # 2. 检查CPU
    cpu_info = platform.processor()
    print(f"CPU: {cpu_info}")
    
    # 3. 检查GPU（使用torch）
    print("\n--- PyTorch检测 ---")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 4. 检查Intel GPU（使用系统命令）
    print("\n--- Intel显卡检测 ---")
    if system == "Windows":
        result = subprocess.run(
            ['wmic', 'path', 'win32_videocontroller', 'get', 'name'],
            capture_output=True, text=True
        )
        if "Intel" in result.stdout:
            print("检测到Intel显卡:")
            for line in result.stdout.split('\n'):
                if "Intel" in line:
                    print(f"  {line.strip()}")
    
    elif system == "Linux":
        result = subprocess.run(
            ['lspci', '|', 'grep', '-i', 'vga'],
            capture_output=True, text=True, shell=True
        )
        if "Intel" in result.stdout:
            print("检测到Intel显卡:")
            print(result.stdout)
    
    # 5. 检查oneAPI支持
    print("\n--- oneAPI支持检测 ---")
    try:
        import intel_extension_for_pytorch as ipex
        print(f"Intel PyTorch扩展已安装: {ipex.__version__}")
        print("✅ 支持Intel XPU编程")
    except ImportError:
        print("❌ Intel PyTorch扩展未安装")
        print("可安装: pip install intel-extension-for-pytorch")


if __name__ == "__main__":
    check_intel_xpu_support()