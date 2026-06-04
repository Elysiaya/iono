import ftplib
import gzip
import os
import shutil
import time
from datetime import datetime, timedelta

from iono.config import Config

def get_ftp_connection(server):
    try:
        ftp = ftplib.FTP_TLS(server, timeout=60)
        ftp.login('anonymous', 'anonymous@example.com')
        ftp.prot_p()
        return ftp
    except Exception as e:
        print(f"重新连接 FTP 失败: {e}")
        return None

def download_ionex_yearly(year, dest_dir=None):
    """
    下载指定年份一整年的电离层地图文件 (IONEX)
    包括自动解压.gz文件，并删除原.gz压缩文件
    """
    server = 'gdc.cddis.eosdis.nasa.gov'
    dest_dir = dest_dir or Config.data_dir / "ionex" / f"ionex_{year}"
    
    # 确保目标文件夹存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print(f"正在连接到服务器 {server} ...")
    ftp = get_ftp_connection(server)
    if not ftp:
        return

    # 设置起止时间：该年的1月1日到12月31日
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    current_date = start_date
    while current_date <= end_date:
        doy = current_date.strftime('%j')    # 提取一年中的第几天 (001 到 366)
        yyyy = current_date.strftime('%Y')   # 提取四位数年份
        yy = current_date.strftime('%y')     # 提取两位数年份
        
        # 构建远程目录和文件名
        dir_path = f"/pub/gps/products/ionex/{yyyy}/{doy}"
        
        # 根据年份选择不同的文件名规则
        if year >= 2022 and int(doy) >= 331:
            filename = f"COD0OPSFIN_{yyyy}{doy}0000_01D_01H_GIM.INX.gz"
            local_comp_path = os.path.join(dest_dir, filename)
            local_unzipped_path = local_comp_path.replace('.gz', '')
        else:
            filename = f"codg{doy}0.{yy}i.Z"
            local_comp_path = os.path.join(dest_dir, filename)
            local_unzipped_path = local_comp_path.replace('.Z', '')
            
        remote_filepath = f"{dir_path}/{filename}"
        
        # 如果解压后的文件或已下载的压缩包已经存在且大小大于0，跳过下载
        if (os.path.exists(local_unzipped_path) and os.path.getsize(local_unzipped_path) > 0) or \
           (os.path.exists(local_comp_path) and os.path.getsize(local_comp_path) > 0):
            print(f"[{yyyy}-{doy}] 文件已存在且有效，跳过。")
            current_date += timedelta(days=1)
            continue

        # 如果存在上次由于中断残留的破损压缩包，先删除
        if os.path.exists(local_comp_path):
            os.remove(local_comp_path)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[{yyyy}-{doy}] 正在下载 {filename}... (尝试 {attempt+1}/{max_retries})")
                # 以二进制模式下载文件
                with open(local_comp_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {remote_filepath}", f.write)
                    
                # 解压文件
                print(f"[{yyyy}-{doy}] 正在解压 {filename}...")
                if filename.endswith('.gz'):
                    with gzip.open(local_comp_path, 'rb') as f_in:
                        with open(local_unzipped_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    # 对于 .Z 文件，Python 原生的 gzip 模块不支持 LZW 算法解压
                    # 这里尝试调用系统的解压命令，如果你安装了 7zip 也可以改用 7z x 命令
                    print("提示: .Z 文件可能需要用到外部工具解压，这里仅保存压缩包（或尝试自行解压）。")
                    # 也可以选择调用命令行: os.system(f'gzip -d "{local_comp_path}"')
                    pass
                        
                # 只有成功解压才删除原压缩文件 (如果是 .Z 文件且没有自行解压，保留它)
                if filename.endswith('.gz'):
                    os.remove(local_comp_path)
                print(f"[{yyyy}-{doy}] 处理完成 -> {local_unzipped_path if filename.endswith('.gz') else local_comp_path}")
                
                # 下载成功，跳出重试循环
                break
                
            except ftplib.error_perm as e:
                print(f"[{yyyy}-{doy}] 服务器上未找到文件或没有权限: {e}")
                if os.path.exists(local_comp_path):
                    os.remove(local_comp_path)
                break # 文件不存在的话重试也没用，直接跳过
            except Exception as e:
                print(f"[{yyyy}-{doy}] 下载或解压时发生错误: {e}")
                if os.path.exists(local_comp_path):
                    os.remove(local_comp_path)
                
                # 如果是网络断开，尝试重新连接
                print("尝试重新连接服务器...")
                try:
                    ftp.quit()
                except:
                    pass
                time.sleep(5) # 暂停 5 秒后重连，避免被服务器拉黑
                ftp = get_ftp_connection(server)
                if not ftp:
                    time.sleep(10)
        
        current_date += timedelta(days=1)
        
    try:
        ftp.quit()
    except:
        pass
    print("全年数据下载任务结束！")

if __name__ == "__main__":
    # 指定需要下载的年份及保存目录
    for year in range(2022, 2023):
        save_directory = Config.data_dir / "ionex" / f"ionex_{year}"
        download_ionex_yearly(year, dest_dir=save_directory)
    # target_year = 2023
    # save_directory = os.path.join("C:\\Users\\zx\\Desktop\\毕业论文\\gim", f"ionex_{target_year}")
    
    # download_ionex_yearly(target_year, dest_dir=save_directory)
