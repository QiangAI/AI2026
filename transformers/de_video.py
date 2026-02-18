import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import time

class RealTimeDPTVisualizer:
    """实时DPT深度估计可视化器"""
    
    def __init__(self, camera_id=0, display_fps=True):
        """
        初始化
        
        Args:
            camera_id: 摄像头ID
            display_fps: 是否显示FPS
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.processor = DPTImageProcessor.from_pretrained("F:/03Models/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("F:/03Models/dpt-large").to(self.device)
        self.model.eval()
        
        self.camera_id = camera_id
        self.display_fps = display_fps
        self.frame_count = 0
        self.fps = 0
        
    def estimate_depth(self, frame):
        """估计深度"""
        # BGR转RGB并转为PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 预处理
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # 上采样
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map = prediction.squeeze().cpu().numpy()
        return depth_map
    
    def process_frame(self, frame):
        """处理单帧并返回可视化结果"""
        # 深度估计
        depth = self.estimate_depth(frame)
        # 归一化并转为彩色图
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_8bit = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        # 并排显示
        h, w = frame.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = frame
        combined[:, w:] = depth_colored
        
        return combined
    
    def run(self):
        """启动实时处理"""
        cap = cv2.VideoCapture(self.camera_id)
        
        # 设置分辨率（可选，降低分辨率可以提高速度）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("按 'q' 退出，按 's' 保存当前帧")
        
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            result = self.process_frame(frame)
            
            # 计算FPS
            if self.display_fps:
                curr_time = time.time()
                self.fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                cv2.putText(result, f"FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, "RGB", (10, result.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result, "Depth", (frame.shape[1] + 10, result.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示
            cv2.imshow("DPT-Large Real-time Depth Estimation", result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                filename = f"dpt_capture_{self.frame_count}.png"
                cv2.imwrite(filename, result)
                print(f"已保存: {filename}")
            
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

class VideoFileDPTVisualizer(RealTimeDPTVisualizer):
    """视频文件DPT深度估计可视化器"""
    
    def __init__(self, video_path, output_path=None):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.out = None
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 初始化输出视频
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                fps, 
                (width * 2, height)
            )
        
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            result = self.process_frame(frame)
            
            # 显示进度
            progress = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * 100)
            cv2.putText(result, f"Progress: {progress}%", (10, result.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示FPS
            curr_time = time.time()
            self.fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(result, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示
            cv2.imshow("DPT-Large Video Depth Estimation", result)
            
            # 写入输出
            if self.out:
                self.out.write(result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        print("处理完成！")

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPT-Large 视频深度估计")
    parser.add_argument("--source", type=str, default="camera",
                       help="输入源: 'camera' 或视频文件路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出视频路径（可选）")
    
    args = parser.parse_args()
    
    if args.source == "camera":
        # 实时摄像头
        visualizer = RealTimeDPTVisualizer(camera_id=0)
        visualizer.run()
    else:
        # 视频文件
        visualizer = VideoFileDPTVisualizer(args.source, args.output)
        visualizer.run()


