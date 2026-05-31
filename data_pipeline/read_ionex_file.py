import numpy as np
import re
from datetime import datetime

def read_ionex_file(filepath):
    """
    读取 IGS IONEX 格式的电离层地图文件，返回 (T, 71, 73) 的 TEC 数组（单位：TECU）
    
    Parameters:
        filepath (str): IONEX 文件路径
    
    Returns:
        tec_maps (np.ndarray): shape (T, 71, 73), dtype=float32
        epochs (list of datetime): 每张图的时间戳
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 查找所有 TEC MAP 块
    map_blocks = []
    current_block = None
    in_map = False

    for line in lines:
        if 'START OF TEC MAP' in line:
            in_map = True
            current_block = {'header': [], 'data_lines': []}
            continue
        elif 'END OF TEC MAP' in line:
            print("Finished reading a TEC MAP block.")
            if current_block is not None:
                map_blocks.append(current_block)
                current_block = None
            in_map = False
            break

        if in_map:
            # 检查是否是纬度行
            if "LAT/LON1/LON2/DLON/H" in line:
                # 这是纬度行，保存
                current_block['data_lines'].append(line.strip())
            else:
                # 可能是数据行（整数）
                stripped = line.strip()
                if stripped and stripped[0].isdigit() or stripped.startswith('-'):
                    current_block['data_lines'].append(stripped)

    # 解析每个 block
    tec_maps = []
    epochs = []

    for block in map_blocks:
        # 提取 epoch（需要从文件头找，但这里简化：假设按顺序）
        # 实际上更稳健的做法是解析 EPOCH 行，但为简化，我们先跳过时间解析
        # 你可以后续用正则匹配 EPOCH 行
        pass

    # 更简单方法：直接扫描所有行，找 EPOCH 和 DATA
    tec_maps = []
    epochs = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "START OF RMS MAP" in line:
            break
        if 'EPOCH OF CURRENT MAP' in line:
            # 解析时间
            parts = line.split()
            year, month, day, hour, minute, second = map(int, parts[:6])
            epoch = datetime(year, month, day, hour, minute, second)
            epochs.append(epoch)
            i += 1
            # 开始读取地图数据
            tec_map = np.zeros((71, 73), dtype=np.float32)
            lat_index = 0
            while i < len(lines) and 'END OF TEC MAP' not in lines[i]:
                line = lines[i].strip()
                if re.match(r'\s*[+-]?\d+\.\d', line):
                    # 纬度行，跳过（我们按顺序读71行）
                    i += 1
                    continue
                # 数据行：空格分隔的整数
                if line and (line[0].isdigit() or line.startswith('-')):
                    # 提取所有整数
                    nums = list(map(int, line.split()))
                    # 当前行属于当前纬度
                    start_col = 0
                    # 计算当前已读多少列
                    cols_so_far = 0
                    row_data = []
                    j = i
                    while cols_so_far < 73 and j < len(lines):
                        sub_line = lines[j].strip()
                        if not sub_line or not (sub_line[0].isdigit() or sub_line.startswith('-')):
                            break
                        sub_nums = list(map(int, sub_line.split()))
                        row_data.extend(sub_nums)
                        cols_so_far += len(sub_nums)
                        j += 1
                    # 取前73个
                    row_vals = np.array(row_data[:73], dtype=np.float32) * 0.1  # 转为 TECU
                    tec_map[lat_index, :] = row_vals
                    lat_index += 1
                    i = j
                else:
                    i += 1
            tec_maps.append(tec_map)
        else:
            i += 1

    tec_maps = np.stack(tec_maps, axis=0)  # (T, 71, 73)
    return tec_maps, epochs
    """
    更健壮地读取 IONEX 文件。
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # 分割成多个 MAP
    map_blocks = content.split('START OF TEC MAP')
    tec_maps = []
    epochs = []

    for block in map_blocks[1:]:  # 第一个是 header
        if 'END OF TEC MAP' not in block:
            continue

        # 提取 EPOCH
        epoch_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*EPOCH OF CURRENT MAP', block)
        if epoch_match:
            y, m, d, H, M, S = map(int, epoch_match.groups())
            epoch = datetime(y, m, d, H, M, S)
            epochs.append(epoch)

        # 提取所有数据行（整数行）
        data_lines = []
        lines = block.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')) and 'LAT' not in line and 'END' not in line:
                try:
                    int(line.split()[0])
                    data_lines.append(line)
                except:
                    pass

        # 合并所有数字
        all_numbers = []
        for line in data_lines:
            nums = list(map(int, line.split()))
            all_numbers.extend(nums)

        # 应该有 71 * 73 = 5183 个值
        if len(all_numbers) != 71 * 73:
            print(f"Warning: expected 5183 values, got {len(all_numbers)}")
            # 尝试截断或填充
            all_numbers = all_numbers[:71*73]
            if len(all_numbers) < 71*73:
                all_numbers.extend([0] * (71*73 - len(all_numbers)))

        tec_map = np.array(all_numbers, dtype=np.float32).reshape(71, 73) * 0.1  # 转为 TECU
        tec_maps.append(tec_map)

    return np.stack(tec_maps, axis=0), epochs


if __name__ == "__main__":
    tec_array, times = read_ionex_file('gim\\COD0OPSFIN_20240010000_01D_01H_GIM.INX')
    print("Shape:", tec_array.shape)  # 应为 (13, 71, 73)