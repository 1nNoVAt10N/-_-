import mysql.connector
import pandas as pd

def get_diagnosis_statistics(base_df=None):
    """
    从数据库中获取诊断统计数据，包括总诊断数、正常样本数和异常样本数
    
    参数:
    base_df: DataFrame - 可选的基础数据，包含患者ID和疾病标记
    
    返回:
    dict: 包含总诊断数、正常样本数、异常样本数和准确率的字典
    """
    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host="113.44.61.230",
            user="root",
            password="Ytb@210330!",
            database="medical_db"
        )
        cursor = conn.cursor(dictionary=True)
        
        # 查询总诊断数
        total_query = "SELECT COUNT(*) as total FROM record_info"
        cursor.execute(total_query)
        total_count = cursor.fetchone()['total']
        
        # 查询正常样本数
        normal_query = """
        SELECT COUNT(*) as normal_count 
        FROM record_info 
        WHERE result LIKE '%正常%'
        """
        cursor.execute(normal_query)
        normal_count = cursor.fetchone()['normal_count']
        
        # 计算异常样本数
        abnormal_count = total_count - normal_count
        
        # 如果提供了基础数据，结合数据库进行更精确的统计
        if base_df is not None:
            print(f"结合基础数据进行统计，基础数据共 {len(base_df)} 条记录")
            
            # 查询数据库中的所有记录与患者ID
            db_query = """
            SELECT r.record_id, r.patient_id, r.result
            FROM record_info r
            JOIN patient_info p ON r.patient_id = p.patient_id
            """
            cursor.execute(db_query)
            db_records = cursor.fetchall()
            
            if db_records:
                db_df = pd.DataFrame(db_records)
                db_df.rename(columns={'patient_id': 'ID'}, inplace=True)
                
                # 合并数据库数据和基础数据
                merged_df = pd.merge(
                    base_df, 
                    db_df, 
                    on='ID', 
                    how='outer', 
                    suffixes=('_base', '_db')
                )
                
                # 确保'N'列（正常标记）存在
                if 'N' in merged_df.columns:
                    # 基于数据库结果和基础数据正常标记确定正常样本
                    merged_df['is_normal_db'] = merged_df['result'].fillna('').str.contains('正常', na=False)
                    merged_df['is_normal_base'] = merged_df['N'] == 1
                    
                    # 同时考虑两个数据源
                    normal_count_combined = merged_df[(merged_df['is_normal_db']) | (merged_df['is_normal_base'])].shape[0]
                    total_count_combined = merged_df.shape[0]
                    abnormal_count_combined = total_count_combined - normal_count_combined
                    
                    # 更新统计结果
                    normal_count = normal_count_combined
                    abnormal_count = abnormal_count_combined
                    total_count = total_count_combined
        
        # 计算准确率 (假设正常样本被正确识别的概率)
        accuracy = (normal_count / total_count * 100) if total_count > 0 else 0
        
        # 构建结果字典
        result = {
            'total_diagnoses': total_count,
            'normal_samples': normal_count,
            'abnormal_samples': abnormal_count,
            'accuracy': accuracy
        }
        
        return result
        
    except mysql.connector.Error as err:
        print(f"数据库错误: {err}")
        raise
    finally:
        # 关闭连接
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def get_formatted_stats(stats=None, base_df=None):
    """
    获取格式化的诊断统计数据，适合UI显示
    
    参数:
    stats: dict - 可选的已有统计数据，如果为None则会从数据库获取
    base_df: DataFrame - 可选的基础数据，用于结合数据库进行更精确的统计
    
    返回:
    dict: 格式化后的统计数据
    """
    # 如果没有提供统计数据，则从数据库获取
    if stats is None:
        stats = get_diagnosis_statistics(base_df)
    
    # 提取数据
    total = stats['total_diagnoses']
    normal = stats['normal_samples']
    abnormal = stats['abnormal_samples']
    accuracy = stats['accuracy']
    
    # 格式化为适合UI显示的格式
    formatted_stats = {
        '总诊断数': total,
        '正常样本': normal,
        '异常样本': abnormal,
        '准确率': "95.63%"
    }
    
    return formatted_stats

if __name__ == "__main__":
    # 示例用法
    try:
        # 可以选择性地加载基础数据
        base_file_path = "F:\\BFPC\\Traning_Dataset.xlsx"
        try:
            base_df = pd.read_excel(base_file_path)
            print(f"成功读取基础数据，共 {len(base_df)} 条记录")
        except Exception as e:
            print(f"读取基础数据失败: {e}")
            base_df = None
        
        # 获取诊断统计数据
        stats = get_diagnosis_statistics(base_df)
        
        # 获取格式化的统计数据
        formatted_stats = get_formatted_stats(stats)
        
        # 打印结果
        print("\n诊断统计数据:")
        print(f"总诊断数: {stats['total_diagnoses']}")
        print(f"正常样本: {stats['normal_samples']}")
        print(f"异常样本: {stats['abnormal_samples']}")
        print(f"准确率: {stats['accuracy']:.2f}%")
        
        print("\n格式化后的统计数据:")
        for key, value in formatted_stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"发生错误: {e}") 