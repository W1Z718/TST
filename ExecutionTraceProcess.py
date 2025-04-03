import os
import sys
import ast
import pandas as pd
import numpy as np
import traceback
import argparse
import tempfile
import threading
import time
from copy import deepcopy
from collections import defaultdict, Counter

# 执行超时异常
class ExecutionTimeoutError(Exception):
    pass

# 代码执行追踪器
class CodeTracer:
    def __init__(self, max_steps=1000, timeout=5):
        self.max_steps = max_steps  # 最大执行步数
        self.timeout = timeout      # 执行超时时间(秒)
        self.traces = []            # 存储轨迹
        self.step_count = 0         # 当前步数
        self.global_vars = {}       # 全局变量状态
        self.local_vars = {}        # 局部变量状态
        self.error = None           # 捕获的错误
        self.error_type = None      # 错误类型
        self.error_line = None      # 错误行号
        
    def trace_function(self, frame, event, arg):
        """每执行一步Python代码时调用的追踪函数"""
        if self.step_count >= self.max_steps:
            raise ExecutionTimeoutError("执行步数超过最大限制")
            
        if event == 'line':
            # 获取当前行号和代码
            lineno = frame.f_lineno
            filename = frame.f_code.co_filename
            function = frame.f_code.co_name
            
            # 只追踪用户代码，忽略库代码
            if '<string>' not in filename and 'tempfile' not in filename:
                return self.trace_function
                
            # 获取当前行的AST节点类型
            source = self._get_source_line(frame)
            op_type = self._get_operation_type(source)
            
            # 获取并复制当前变量状态(深拷贝避免引用问题)
            local_vars = {}
            for name, value in frame.f_locals.items():
                # 只记录基本类型，避免复杂对象
                if isinstance(value, (int, float, str, bool)) or value is None:
                    local_vars[name] = deepcopy(value)
                else:
                    try:
                        local_vars[name] = str(type(value))
                    except:
                        local_vars[name] = "complex_object"
            
            # 记录这一步的执行信息
            trace_step = {
                'step': self.step_count,
                'event': event,
                'lineno': lineno,
                'function': function,
                'source': source,
                'op_type': op_type,
                'local_vars': local_vars
            }
            
            self.traces.append(trace_step)
            self.step_count += 1
            self.local_vars = local_vars
                
        elif event == 'exception':
            exc_type, exc_value, exc_traceback = arg
            self.error = exc_value
            self.error_type = exc_type.__name__
            self.error_line = frame.f_lineno
            
            # 记录异常信息
            trace_step = {
                'step': self.step_count,
                'event': 'exception',
                'lineno': frame.f_lineno,
                'error_type': self.error_type,
                'error_msg': str(exc_value)
            }
            self.traces.append(trace_step)
            
        return self.trace_function
        
    def _get_source_line(self, frame):
        """获取当前行的源代码"""
        try:
            return frame.f_code.co_filename
        except:
            return "unknown"
            
    def _get_operation_type(self, source):
        """推断当前行的操作类型"""
        # 在实际实现中，可以尝试解析AST判断操作类型
        # 这里简化处理
        return "unknown"
        
    def run_with_trace(self, code_str):
        """使用追踪器运行代码"""
        self.traces = []
        self.step_count = 0
        self.error = None
        
        # 创建一个线程运行代码，以便支持超时终止
        def execute_code():
            try:
                # 设置追踪器
                sys.settrace(self.trace_function)
                
                # 创建临时模块执行代码
                temp_module = {'__name__': '__main__'}
                
                # 执行代码
                exec(code_str, temp_module)
                
                # 保存全局变量
                self.global_vars = {k: v for k, v in temp_module.items() 
                                   if not k.startswith('__')}
                                   
            except ExecutionTimeoutError as e:
                self.error = e
                self.error_type = "TimeoutError"
            except Exception as e:
                # 捕获所有其他异常
                self.error = e
                self.error_type = type(e).__name__
            finally:
                # 恢复追踪器
                sys.settrace(None)
        
        # 创建并启动线程
        thread = threading.Thread(target=execute_code)
        thread.daemon = True
        thread.start()
        
        # 等待执行完成或超时
        thread.join(self.timeout)
        
        # 如果线程还在运行，说明超时了
        if thread.is_alive():
            self.error = ExecutionTimeoutError("执行超时")
            self.error_type = "TimeoutError"
            # 强制停止追踪
            sys.settrace(None)
        
        return self.traces
        
    def trace(self, code_file):
        """追踪代码文件执行"""
        try:
            with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                code_str = f.read()
            return self.run_with_trace(code_str)
        except Exception as e:
            # 文件读取或其他错误
            self.error = e
            self.error_type = type(e).__name__
            return []

# 提取执行轨迹特征
def extract_execution_features(traces, max_seq_len=100):
    """从执行轨迹中提取特征序列"""
    # 确保输出维度为 [max_seq_len, feature_dim]
    feature_dim = 10  # 或者根据需要调整
    features = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
    
    # 填充特征
    for i, trace in enumerate(traces[:max_seq_len]):
        if i >= max_seq_len:
            break
            
        # 提取每一步的基本特征
        step_features = [
            trace.get('step', 0),                          # 步骤编号
            trace.get('lineno', 0),                        # 行号
            float(trace.get('event', '') == 'line'),       # 是否是行执行事件
            float(trace.get('event', '') == 'exception'),  # 是否是异常事件
            # ... 其他特征 ...
        ]
        
        # 可以添加变量状态特征
        local_vars = trace.get('local_vars', {})
        var_count = len(local_vars)
        step_features.extend([
            var_count,                                     # 变量数量
            # ... 可以添加更多特征 ...
        ])
        
        # 确保特征维度正确
        features[i, :len(step_features)] = step_features[:feature_dim]
    
    return features

# 创建执行轨迹时间序列数据集
def create_execution_trace_dataset(code_folder, label_file, output_folder, max_seq_len=100):
    """处理代码文件并生成.ts格式的执行轨迹特征时间序列数据"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取标签文件
    labels_df = pd.read_excel(label_file)
    
    # 准备输出 - 修正元数据格式，添加必要的字段
    ts_lines = [
        "@problemname 3004_execution_trace",
        "@timestamps false",         # 添加必需的timestamps字段
        "@univariate false",         # 添加必需的univariate字段
        "@timestamp 2024-03-31",
        "@sampleSize 785",          
        "@targetlabel intlabel",
        "@attributeDefinition",
        "@classname intlabel",
        "@classlabel true 0 1 2 3 4 5 6 7",  # 8个分类类别
        "@data"
    ]
    
    tracer = CodeTracer(max_steps=1000, timeout=10)
    
    successful_files = 0
    failed_files = 0
    
    # 处理每个代码文件
    for index, row in labels_df.iterrows():
        filename = row['filename']
        label = row['intlabel']
        
        file_path = os.path.join(code_folder, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 {filename} 不存在，已跳过")
            failed_files += 1
            continue
        
        try:
            print(f"正在处理 {filename}...")
            
            # 追踪代码执行
            traces = tracer.trace(file_path)
            
            # 提取执行特征
            features = extract_execution_features(traces, max_seq_len)
            
            # 为错误代码提取额外特征
            error_features = np.zeros(10)
            if tracer.error:
                error_features[0] = 1  # 有错误标记
                # 错误类型编码
                error_codes = {
                    'NameError': 1, 'TypeError': 2, 'ValueError': 3, 
                    'IndexError': 4, 'ZeroDivisionError': 5, 'TimeoutError': 6
                }
                error_features[1] = error_codes.get(tracer.error_type, 7)
                error_features[2] = tracer.error_line if tracer.error_line else -1
            
            # 构建.ts格式的数据行
            line_parts = []
            
            # 处理时间序列特征
            for dim in range(features.shape[1]):
                dim_values = features[:, dim]
                dim_str = ':'.join([str(v) for v in dim_values])
                line_parts.append(dim_str)
            
            # 添加错误特征作为额外维度
            error_stats_str = ':'.join([str(v) for v in error_features])
            line_parts.append(error_stats_str)
            
            # 添加标签
            ts_line = ':'.join(line_parts) + ':' + str(label)
            ts_lines.append(ts_line)
            
            successful_files += 1
            
        except Exception as e:
            print(f"错误: 处理文件 {filename} 时出错: {str(e)}")
            failed_files += 1
            continue
    
    # 写入.ts文件
    output_path = os.path.join(output_folder, "3004_execution_trace.ts")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ts_lines))
    
    print(f"处理完成. 成功: {successful_files}, 失败: {failed_files}")
    print(f"输出文件保存在: {output_path}")
    
    # 生成并运行数据类注册脚本
    register_script = """\"\"\"
注册3004_execution_trace数据类到数据工厂
\"\"\"

import os
import sys

# 获取主代码路径并添加到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # 使用正确的导入路径
    from src.datasets.data import TSClassificationArchive, data_factory
    
    # 扩展数据工厂，添加3004_execution_trace数据类
    class Code3004ExecutionTraceData(TSClassificationArchive):
        \"\"\"3004代码执行轨迹数据\"\"\"
        
        def __init__(self, root_dir, file_list=None, pattern=None, **kwargs):
            # 使用与父类相同的初始化参数，但指定不同的数据文件
            super().__init__(root_dir, file_list=['3004_execution_trace.ts'], **kwargs)
    
    # 注册新的数据类
    data_factory['3004_execution_trace'] = Code3004ExecutionTraceData
    
    print("成功注册 3004_execution_trace 数据类")
    print("现在支持的数据类:")
    for key in sorted(data_factory.keys()):
        print(f" - {key}")
    
except ImportError as e:
    print(f"导入数据工厂失败: {e}")
    print("\\n解决方法:")
    print("1. 确保当前目录是项目根目录（包含src文件夹的目录）")
    print("2. 尝试添加src目录到PYTHONPATH:")
    print("   export PYTHONPATH=$PYTHONPATH:$(pwd)  # Linux/Mac")
    print("   set PYTHONPATH=%PYTHONPATH%;%cd%      # Windows")
    sys.exit(1)
except Exception as e:
    print(f"注册数据类时出错: {e}")
    sys.exit(1)
"""
    with open('register_data_class.py', 'w', encoding='utf-8') as f:
        f.write(register_script)
    
    print("\n自动生成了数据类注册脚本. 请在运行主程序前执行:")
    print("python register_data_class.py")
    
    return output_path

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将代码文件转换为执行轨迹时间序列数据")
    parser.add_argument("--code_folder", default="3004/3004_code", help="代码文件夹路径")
    parser.add_argument("--label_file", default="3004/3004_label.xlsx", help="标签文件路径")
    parser.add_argument("--output_folder", default="data/SelectedData", help="输出文件夹路径")
    parser.add_argument("--max_seq_len", type=int, default=100, help="每个样本的最大序列长度")
    
    args = parser.parse_args()
    
    ts_file = create_execution_trace_dataset(
        args.code_folder, 
        args.label_file, 
        args.output_folder, 
        args.max_seq_len
    )
    
    print(f"\n要运行训练，请使用以下命令:")
    print(f"python main.py --output_dir experiments --data_dir {args.output_folder} --data_class 3004_execution_trace --task classification --epochs 100 --batch_size 32 --d_model 256 --num_heads 8 --num_layers 3 --max_seq_len {args.max_seq_len}") 