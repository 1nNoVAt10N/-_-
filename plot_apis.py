import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_disease_age_distribution(base_df, disease_type):
    """
    将基础数据与数据库数据结合，为指定疾病类型绘制年龄分布饼图
    
    参数:
    base_df: DataFrame - 基础数据，包含患者ID、年龄、性别和疾病标记
    disease_type: str - 要分析的疾病类型代码，例如 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'N'
    
    返回:
    tuple: (fig, age_percentages) - Matplotlib图像对象和疾病年龄分布百分比
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
            return None, None
        
        # 定义年龄分组
        age_bins = [0, 18, 30, 45, 60, 75, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
        
        # 创建年龄组
        disease_df['age_group'] = pd.cut(disease_df['Patient Age'], bins=age_bins, labels=age_labels)
        
        # 统计各年龄组的患者数量
        age_counts = disease_df['age_group'].value_counts().sort_index()
        
        # 设置饼图颜色
        colors = plt.cm.Spectral(np.linspace(0, 1, len(age_counts)))
        
        # 创建图形对象
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建饼图
        wedges, texts, autotexts = ax.pie(
            age_counts,
            labels=age_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            shadow=False,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # 设置文本样式
        plt.setp(autotexts, size=10, weight='bold')
        plt.setp(texts, size=12)
        
        # 添加标题
        ax.set_title(f'{disease_name}患者的年龄分布', fontsize=16, pad=20)
        
        # 添加图例
        ax.legend(
            wedges,
            [f"{age}: {count}人" for age, count in zip(age_counts.index, age_counts.values)],
            title="年龄组(人数)",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        ax.axis('equal')  # 保持饼图为圆形
        fig.tight_layout()
        
        # 计算年龄分布百分比
        age_percentages = age_counts / age_counts.sum() * 100
        
        # 返回图形对象和年龄分布百分比
        return fig, age_percentages
        
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
    将基础数据与数据库数据结合，为指定疾病类型绘制性别分布饼图
    
    参数:
    base_df: DataFrame - 基础数据，包含患者ID、年龄、性别和疾病标记
    disease_type: str - 要分析的疾病类型代码，例如 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'N'
    
    返回:
    tuple: (fig, gender_percentages) - Matplotlib图像对象和疾病性别分布百分比
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
            return None, None
        
        # 统计各性别的患者数量
        gender_counts = disease_df['Gender'].value_counts()
        
        # 设置颜色
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # 蓝、红、绿、橙
        
        # 创建图形对象
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建饼图
        wedges, texts, autotexts = ax.pie(
            gender_counts,
            labels=gender_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(gender_counts)],
            shadow=False,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # 设置文本样式
        plt.setp(autotexts, size=10, weight='bold')
        plt.setp(texts, size=12)
        
        # 添加标题
        ax.set_title(f'{disease_name}患者的性别分布', fontsize=16, pad=20)
        
        # 添加图例
        ax.legend(
            wedges,
            [f"{gender}: {count}人" for gender, count in zip(gender_counts.index, gender_counts.values)],
            title="性别(人数)",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        ax.axis('equal')  # 保持饼图为圆形
        fig.tight_layout()
        
        # 计算性别分布百分比
        gender_percentages = gender_counts / gender_counts.sum() * 100
        
        # 返回图形对象和性别分布百分比
        return fig, gender_percentages
        
    except mysql.connector.Error as err:
        print(f"数据库错误: {err}")
        raise
    finally:
        # 关闭连接
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()



def main(file_path, db_analysis=True):
    """
    主函数：加载基础数据，并与数据库数据结合进行分析
    
    参数:
    file_path: str - Excel文件路径
    db_analysis: bool - 是否进行数据库分析
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
        
        fig,_ = plot_disease_age_distribution(base_df, 'N')
        fig2,_ = plot_disease_gender_distribution(base_df, 'N')
        return fig,fig2
    except Exception as e:
        print(f"错误: {e}")
        raise

# 如果作为主程序运行
if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 示例用法
    file_path = "F:\\BFPC\\Traning_Dataset.xlsx"  # 基础数据文件路径
    _,x = main(file_path)
    print(x)