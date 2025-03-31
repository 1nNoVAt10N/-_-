import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_disease_age_distribution(base_df, disease_type):
    """
    将基础数据与数据库数据结合，计算指定疾病类型的年龄分布百分比
    
    参数:
    base_df: DataFrame - 基础数据，包含患者ID、年龄、性别和疾病标记
    disease_type: str - 要分析的疾病类型代码，例如 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'N'
    
    返回:
    age_percentages: Series - 疾病年龄分布百分比
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
        
        # 疾病代码到疾病名称的映射
        disease_mapping = {
            'N': '正常',
            'D': '糖尿病视网膜病变',
            'G': '青光眼',
            'C': '白内障',
            'A': '年龄相关性黄斑变性',
            'H': '高血压',
            'M': '近视',
            'O': '其他疾病'
        }
        
        # 获取疾病全名
        disease_name = disease_mapping.get(disease_type, disease_type)
        
        # 从数据库查询与该疾病相关的记录
        query = """
        SELECT r.record_id, r.result, p.patient_id, p.patient_age, p.patient_gender
        FROM record_info r
        JOIN patient_info p ON r.patient_id = p.patient_id
        WHERE r.result LIKE %s
        """
        
        search_term = f"%{disease_name}%"
        cursor.execute(query, (search_term,))
        db_records = cursor.fetchall()
        
        # 将数据库记录转换为DataFrame
        if db_records:
            db_df = pd.DataFrame(db_records)
            print(f"从数据库找到 {len(db_df)} 条 '{disease_name}' 相关记录")
            
            # 基于patient_id合并数据库数据和基础数据
            db_df.rename(columns={'patient_id': 'ID'}, inplace=True)
            merged_df = pd.merge(
                base_df, 
                db_df, 
                on='ID', 
                how='outer', 
                suffixes=('_base', '_db')
            )
            
            # 年龄处理：优先使用基础数据的年龄，否则使用数据库数据
            merged_df['Patient Age'] = merged_df['Patient Age'].fillna(merged_df['patient_age'])
            
            # 筛选有指定疾病的数据
            # 方法1：使用基础数据中的疾病标记
            condition1 = merged_df[disease_type] == 1
            # 方法2：使用数据库中的结果字段
            condition2 = merged_df['result'].fillna('').str.contains(disease_name, na=False)
            
            # 合并两个条件 (患者在任一数据源中有该疾病)
            disease_df = merged_df[condition1 | condition2]
        else:
            print(f"数据库中没有 '{disease_name}' 相关记录，仅使用基础数据")
            # 仅使用基础数据中标记为有该疾病的患者
            disease_df = base_df[base_df[disease_type] == 1]
        
        # 如果没有找到患者数据，返回
        if len(disease_df) == 0:
            print(f"没有找到 '{disease_name}' 的患者记录")
            return None
        
        # 定义年龄分组
        age_bins = [0, 18, 30, 45, 60, 75, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
        
        # 创建年龄组
        disease_df['age_group'] = pd.cut(disease_df['Patient Age'], bins=age_bins, labels=age_labels)
        
        # 统计各年龄组的患者数量
        age_counts = disease_df['age_group'].value_counts().sort_index()
        
        # 计算年龄分布百分比
        age_percentages = age_counts / age_counts.sum() * 100
        
        # 返回年龄分布百分比
        return age_percentages
        
    except mysql.connector.Error as err:
        print(f"数据库错误: {err}")
        raise
    finally:
        # 关闭连接
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()



def plot_disease_gender_distribution(base_df, disease_type):
    """
    将基础数据与数据库数据结合，计算指定疾病类型的性别分布百分比
    
    参数:
    base_df: DataFrame - 基础数据，包含患者ID、年龄、性别和疾病标记
    disease_type: str - 要分析的疾病类型代码，例如 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'N'
    
    返回:
    gender_percentages: Series - 疾病性别分布百分比
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
        
        # 疾病代码到疾病名称的映射
        disease_mapping = {
            'N': '正常',
            'D': '糖尿病视网膜病变',
            'G': '青光眼',
            'C': '白内障',
            'A': '年龄相关性黄斑变性',
            'H': '高血压',
            'M': '近视',
            'O': '其他疾病'
        }
        
        # 获取疾病全名
        disease_name = disease_mapping.get(disease_type, disease_type)
        
        # 从数据库查询与该疾病相关的记录
        query = """
        SELECT r.record_id, r.result, p.patient_id, p.patient_age, p.patient_gender
        FROM record_info r
        JOIN patient_info p ON r.patient_id = p.patient_id
        WHERE r.result LIKE %s
        """
        
        search_term = f"%{disease_name}%"
        cursor.execute(query, (search_term,))
        db_records = cursor.fetchall()
        
        # 将数据库记录转换为DataFrame
        if db_records:
            db_df = pd.DataFrame(db_records)
            print(f"从数据库找到 {len(db_df)} 条 '{disease_name}' 相关记录")
            
            # 基于patient_id合并数据库数据和基础数据
            db_df.rename(columns={'patient_id': 'ID'}, inplace=True)
            merged_df = pd.merge(
                base_df, 
                db_df, 
                on='ID', 
                how='outer', 
                suffixes=('_base', '_db')
            )
            
            # 性别处理：优先使用基础数据的性别，否则使用数据库数据
            # 确保Patient Sex列存在
            if 'Patient Sex' in merged_df.columns:
                merged_df['Gender'] = merged_df['Patient Sex']
            else:
                merged_df['Gender'] = None
                
            # 如果Gender列为空，使用patient_gender列
            merged_df['Gender'] = merged_df['Gender'].fillna(merged_df['patient_gender'])
            
            # 筛选有指定疾病的数据
            # 方法1：使用基础数据中的疾病标记
            condition1 = merged_df[disease_type] == 1
            # 方法2：使用数据库中的结果字段
            condition2 = merged_df['result'].fillna('').str.contains(disease_name, na=False)
            
            # 合并两个条件 (患者在任一数据源中有该疾病)
            disease_df = merged_df[condition1 | condition2]
        else:
            print(f"数据库中没有 '{disease_name}' 相关记录，仅使用基础数据")
            # 仅使用基础数据中标记为有该疾病的患者
            disease_df = base_df[base_df[disease_type] == 1].copy()
            disease_df['Gender'] = disease_df['Patient Sex']
        
        # 如果没有找到患者数据，返回
        if len(disease_df) == 0:
            print(f"没有找到 '{disease_name}' 的患者记录")
            return None
        
        # 统计各性别的患者数量
        gender_counts = disease_df['Gender'].value_counts()
        
        # 计算性别分布百分比
        gender_percentages = gender_counts / gender_counts.sum() * 100
        
        # 返回性别分布百分比
        return gender_percentages
        
    except mysql.connector.Error as err:
        print(f"数据库错误: {err}")
        raise
    finally:
        # 关闭连接
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()



def get_diagnostic_statistics(base_df=None):
    """
    从数据库中读取诊断统计数据，包括总诊断数、正常样本数和异常样本数
    
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
        # 这里的准确率计算可能需要根据实际需求调整
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

def format_diagnostic_statistics(stats):
    """
    将诊断统计数据格式化为UI显示所需的格式
    
    参数:
    stats: dict - 包含总诊断数、正常样本数、异常样本数和准确率的字典
    
    返回:
    dict: 格式化后的统计数据，适合UI显示
    """
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
        '准确率': f"{accuracy:.2f}%"
    }
    
    return formatted_stats

# 修改main函数，添加格式化选项
def main(file_path, db_analysis=True, statistics=False, format_stats=False, flag="N"):
    """
    主函数：加载基础数据，并与数据库数据结合进行分析
    
    参数:
    file_path: str - Excel文件路径
    db_analysis: bool - 是否进行数据库分析
    statistics: bool - 是否获取诊断统计数据
    format_stats: bool - 是否将统计数据格式化为UI显示所需格式
    
    返回:
    tuple或dict: 根据参数返回不同类型的结果
    """
    # 导入基础数据
    try:
        print(f"读取基础数据: {file_path}")
        base_df = pd.read_excel(file_path)
        print(f"成功读取基础数据，共 {len(base_df)} 条记录")
        
        # 基础数据预处理
        # 1. 确保列名符合预期
        expected_columns = ['ID', 'Patient Age', 'Patient Sex']
        disease_codes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        
        # 检查基本列是否存在
        missing_cols = [col for col in expected_columns if col not in base_df.columns]
        if missing_cols:
            print(f"警告: 基础数据缺少以下列: {missing_cols}")
        
        # 检查疾病代码列是否存在
        missing_disease_cols = [col for col in disease_codes if col not in base_df.columns]
        if missing_disease_cols:
            print(f"警告: 基础数据缺少以下疾病代码列: {missing_disease_cols}")
            print("将为缺失的疾病代码列创建并填充为0")
            for col in missing_disease_cols:
                base_df[col] = 0
        
        # 根据参数执行不同的分析
        if statistics:
            # 获取诊断统计数据
            stats = get_diagnostic_statistics(base_df)
            # 如果需要格式化，则返回格式化后的数据
            if format_stats:
                return format_diagnostic_statistics(stats)
            return stats
        else:
            # 执行原有的年龄和性别分布分析
            age_percentages = plot_disease_age_distribution(base_df, flag)
            gender_percentages = plot_disease_gender_distribution(base_df, flag)
            return age_percentages, gender_percentages
    except Exception as e:
        print(f"错误: {e}")
        raise

# 如果作为主程序运行
if __name__ == "__main__":
    # 示例用法
    file_path = "F:\\BFPC\\Traning_Dataset.xlsx"  # 基础数据文件路径
    
    # 获取统计数据并格式化
    formatted_stats = main(file_path, statistics=True, format_stats=True)
    print("\n格式化后的诊断统计数据:")
    for key, value in formatted_stats.items():
        print(f"{key}: {value}")
    
    # 获取原始统计数据
    stats = main(file_path, statistics=True)
    print("\n原始诊断统计数据:")
    print(f"总诊断数: {stats['total_diagnoses']}")
    print(f"正常样本: {stats['normal_samples']}")
    print(f"异常样本: {stats['abnormal_samples']}")
    print(f"准确率: {stats['accuracy']:.2f}%")
    
    # 获取分布数据
    age_pct, gender_pct = main(file_path)
    print("\n年龄分布百分比:")
    print(age_pct)
    print("\n性别分布百分比:")
    print(gender_pct)
