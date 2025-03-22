import mysql.connector
from datetime import datetime
import base64
def make_user_info(
        user_id=None,
        user_account=None,
        user_gender=None,
        user_age=None,
        user_phone=None,
        user_email=None,
        user_password="123456",):
    conn = mysql.connector.connect(
        host="113.44.61.230",
        user="root",
        password="Ytb@210330!",
        database="medical_db"
    )
    cursor = conn.cursor()


    # 插入 SQL 语句
    query = """
    INSERT INTO user_info (user_id, user_account,user_password,email,phone_num,user_gender,user_age)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    # 插入数据，处理 None 值
    cursor.execute(query, (user_id, user_account, user_password, user_email, user_phone, user_gender, user_age ))
    
    # 提交事务
    conn.commit()

    # 关闭连接
    cursor.close()
    conn.close()


def insert_fund_info(
        patient_id, 
        left_fund=None, 
        left_fund_keyword=None, 
        right_fund=None, 
        right_fund_keyword=None):
    """
    向影像信息表(fund_info)插入眼底影像数据
    
    参数:
    patient_id (int): 患者ID，必填
    left_fund (str/bytes): 左眼眼底图像的base64编码或二进制数据，可选
    left_fund_keyword (str): 左眼眼底关键词，可选
    right_fund (str/bytes): 右眼眼底图像的base64编码或二进制数据，可选
    right_fund_keyword (str): 右眼眼底关键词，可选
    
    返回:
    int: 新插入记录的fund_id
    """
    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host="113.44.61.230",
            user="root",
            password="Ytb@210330!",
            database="medical_db"
        )
        cursor = conn.cursor()
        
        # 获取新的fund_id
        cursor.execute("SELECT MAX(fund_id) FROM fund_info")
        max_fund_id = cursor.fetchone()[0]
        new_fund_id = 1 if max_fund_id is None else max_fund_id + 1
        
        # 准备二进制数据
        left_fund_binary = None
        if left_fund:
            # 如果是字符串格式的base64，转换为二进制
            if isinstance(left_fund, str):
                left_fund_binary = base64.b64decode(left_fund)
            else:
                left_fund_binary = left_fund
                
        right_fund_binary = None
        if right_fund:
            # 如果是字符串格式的base64，转换为二进制
            if isinstance(right_fund, str):
                right_fund_binary = base64.b64decode(right_fund)
            else:
                right_fund_binary = right_fund
        
        # 插入SQL语句
        query = """
        INSERT INTO fund_info 
        (fund_id, left_fund_keyword, left_fund, right_fund_keyword, right_fund, patient_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        # 执行插入
        cursor.execute(query, (
            new_fund_id,
            left_fund_keyword,
            left_fund_binary,
            right_fund_keyword,
            right_fund_binary,
            patient_id
        ))
        
        # 提交事务
        conn.commit()
        
        print(f"影像信息已插入: fund_id={new_fund_id}, patient_id={patient_id}")
        
        return new_fund_id
        
    except mysql.connector.Error as err:
        print(f"数据库错误: {err}")
        if conn.is_connected():
            conn.rollback()
        raise
    finally:
        # 关闭连接
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

# 示例用法
def read_image_file(file_path):
    """读取图像文件并返回二进制数据"""
    try:
        with open(file_path, 'rb') as file:
            return file.read()
    except Exception as e:
        print(f"读取图像文件失败: {e}")
        return None

def make_patend_info(
        patient_id=None,
        patient_name="张三",
        patient_age=None,
        patient_sex=None):
    
    # 连接数据库
    conn = mysql.connector.connect(
        host="113.44.61.230",
        user="root",
        password="Ytb@210330!",
        database="medical_db"
    )
    cursor = conn.cursor()

    # 如果 patient_id 为空，则自动生成新的 patient_id
    if patient_id is None:
        cursor.execute("SELECT MAX(patient_id) FROM patient_info")
        max_patient_id = cursor.fetchone()[0]
        patient_id = 1 if max_patient_id is None else max_patient_id + 1

    # 插入 SQL 语句
    query = """
    INSERT INTO patient_info (patient_id, patient_name,patient_gender, patient_age)
    VALUES (%s, %s, %s, %s)
    """

    # 插入数据，处理 None 值
    cursor.execute(query, (patient_id, patient_name, patient_sex or "Other",patient_age or 0 ))
    
    # 提交事务
    conn.commit()
    
    print(f"患者信息已插入: ID={patient_id}, 姓名={patient_name}, 年龄={patient_age or '未知'}, 性别={patient_sex or '未知'}")

    # 关闭连接
    cursor.close()
    conn.close()

    
def save_results(patient_id=None,
                 patient_name="张三",
                 patient_age=None,
                 patient_sex=None,
                 predict_result=None,
                 advise=None,
                 fund_id=None,
                 left_fund=None,
                 left_fund_keyword=None,
                 right_fund=None,
                 right_fund_keyword=None):
   
    user_id = 1

    conn = mysql.connector.connect(
    host = "113.44.61.230",
    user = "root",
    password = "Ytb@210330!",
    database = "medical_db" )
    cursor = conn.cursor()
    
    # 首先检查患者ID是否存在
    if patient_id is not None:
        check_query = "SELECT COUNT(*) FROM patient_info WHERE patient_id = %s"
        cursor.execute(check_query, (patient_id,))
        exists = cursor.fetchone()[0]
        
        # 如果患者不存在，先插入患者信息
        if exists == 0:
            make_patend_info(
                patient_id=patient_id,
                patient_name=patient_name,
                patient_age=patient_age,
                patient_sex=patient_sex
            )
    else:
        # 如果没有提供patient_id，则创建新患者并获取新生成的ID
        make_patend_info(
            patient_id=None,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_sex=patient_sex
        )
        
        # 获取刚刚创建的患者ID
        cursor.execute("SELECT MAX(patient_id) FROM patient_info")
        patient_id = cursor.fetchone()[0]

    # 检查是否提供了fund_id
    if fund_id is None:
        # 如果提供了任何眼底图像相关数据，则插入fund_info
        if left_fund is not None or right_fund is not None or left_fund_keyword is not None or right_fund_keyword is not None:
            try:
                # 调用insert_fund_info函数插入眼底图像数据
                fund_id = insert_fund_info(
                    patient_id=patient_id,
                    left_fund=left_fund,
                    left_fund_keyword=left_fund_keyword,
                    right_fund=right_fund,
                    right_fund_keyword=right_fund_keyword
                )
                print(f"已自动创建fund_id: {fund_id}")
            except Exception as e:
                print(f"插入眼底图像数据失败: {e}")
                # 如果插入失败但不影响主要功能，可以继续执行
                fund_id = None
        else:
            # 如果没有提供眼底图像数据，则fund_id保持为None
            print("未提供眼底图像数据，fund_id将为NULL")

    ask_recordid_query = "SELECT MAX(record_id) FROM record_info"
    cursor.execute(ask_recordid_query)
    max_record_id = cursor.fetchone()[0]

    # 处理空值情况，表为空时从1开始
    now_record_id = 1 if max_record_id is None else max_record_id + 1

    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 获取当前时间

    # SQL 语句
    query = """
        INSERT INTO record_info (record_id, user_id, patient_id, fund_id, result, suggestion, diagnosis_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

    # 执行插入
    cursor.execute(query, (now_record_id, user_id, patient_id, fund_id, predict_result, advise, now_time))
    conn.commit()  # 提交事务
    cursor.close()
    conn.close()
    print("数据插入成功")
    return now_record_id, fund_id

if __name__ == "__main__":
    # make_user_info(user_id=1,
    #                user_account="123456",
    #                user_password="123456",
    #                user_email="EMAIL",
    #                user_phone="1234567890",
    #                user_gender="Male",
    #                user_age=20)
    left_fund = read_image_file("F:\BFPC\cropped_#Training_Dataset/1_left.jpg")
    right_fund = read_image_file("F:\BFPC\cropped_#Training_Dataset/1_right.jpg")
    save_results(patient_id=1,predict_result="糖尿病，青光眼",advise="不要吃什么",patient_age=20,patient_sex='Male',
                 left_fund_keyword="left",
                 right_fund_keyword="right",
                 left_fund=left_fund,
                 right_fund=right_fund
                 )
