import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from iono.config import Config
from iono.model_fgl import TeacherForecaster

def test_load_teacher():
    """测试实例化并加载教师模型，打印基本信息"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- 开始测试加载教师模型 ---")
    try:
        # 1. 实例化教师模型
        teacher = TeacherForecaster(
            in_channels=Config.in_channels,
            hidden_channels=Config.hidden_channels,
            num_layers=Config.num_layers,
            num_aux=5,
            pred_steps=Config.pred_steps,
            priv_gru_hidden=Config.priv_gru_hidden
        ).to(device)
        
        total_params = sum(p.numel() for p in teacher.parameters())
        print(f"✅ 模型实例化成功！总参数量: {total_params:,}")
        
        # 2. 尝试加载 Checkpoint
        ckpt_path = Config.teacher_checkpoint
        print(f"尝试读取 Checkpoint 路径: {ckpt_path}")
        
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            state_dict = ckpt.get('model_state_dict', ckpt)
            
            # 处理 DataParallel 保存时带有的 'module.' 前缀
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            teacher.load_state_dict(new_state_dict)
            print(f"✅ 成功加载 Checkpoint 权重！")
            
            if 'epoch' in ckpt: print(f"   - 保存的 Epoch: {ckpt['epoch']}")
            if 'train_loss' in ckpt: print(f"   - 训练集 Loss: {ckpt['train_loss']:.6f}")
            if 'val_loss' in ckpt: print(f"   - 验证集 Loss: {ckpt['val_loss']:.6f}")
        else:
            print(f"⚠️ 未找到预训练 Checkpoint 文件，仅使用随机初始化。")
    except Exception as e:
        print(f"❌ 加载失败，发生异常: {e}")
        
    print("--- 教师模型测试结束 ---\n")

if __name__ == "__main__":
    test_load_teacher()
