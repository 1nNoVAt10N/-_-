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
    print(patient_id)
    # 如果 patient_id 为空，则自动生成新的 patient_id
    if patient_id is None and patient_id != 0:
        cursor.execute("SELECT MAX(patient_id) FROM patient_info")
        max_patient_id = cursor.fetchone()[0]
        patient_id = 1 if max_patient_id is None else max_patient_id + 1
    print(patient_id)
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

def get_recent_records(limit=5):
    """
    查询最近插入诊断信息表中的最新记录，并查询相关的患者信息和影像信息
    
    参数:
    limit (int): 返回的记录数量，默认为5条
    
    返回:
    list: 包含结构化数据的记录列表
    """
    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host="113.44.61.230",
            user="root",
            password="Ytb@210330!",
            database="medical_db"
        )
        cursor = conn.cursor(dictionary=True)  # 使用字典游标获取结果
        
        # 查询最近的诊断记录
        query = """
        SELECT r.record_id, r.patient_id, r.fund_id, r.result, r.suggestion, r.diagnosis_date, r.user_id
        FROM record_info r
        ORDER BY r.diagnosis_date DESC
        LIMIT %s
        """
        cursor.execute(query, (limit,))
        records = cursor.fetchall()
        
        # 查询患者信息和影像信息
        result_records = []
        
        for record in records:
            patient_id = record['patient_id']
            fund_id = record['fund_id']

            # 查询患者信息
            patient_query = """
            SELECT patient_id, patient_name, patient_gender, patient_age
            FROM patient_info
            WHERE patient_id = %s
            """
            cursor.execute(patient_query, (patient_id,))
            patient_info = cursor.fetchone()
            
            # 查询影像信息（包含二进制数据）
            fund_info = None
            if fund_id is not None:
                fund_query = """
                SELECT fund_id, left_fund_keyword, right_fund_keyword, patient_id, left_fund, right_fund
                FROM fund_info
                WHERE fund_id = %s
                """
                cursor.execute(fund_query, (fund_id,))
                fund_info = cursor.fetchone()
                
                # 如果存在二进制数据，将其转换为base64编码字符串
                if fund_info:
                    if fund_info['left_fund'] is not None:
                        fund_info['left_fund'] = base64.b64encode(fund_info['left_fund']).decode('utf-8')
                    if fund_info['right_fund'] is not None:
                        fund_info['right_fund'] = base64.b64encode(fund_info['right_fund']).decode('utf-8')
            
            # 构建结构化记录
            structured_record = {
                'record': record,
                'patient': patient_info,
                'fund': fund_info
            }
            
            result_records.append(structured_record)
        
        return result_records
        
    except mysql.connector.Error as err:
        print(f"数据库错误: {err}")
        raise
    finally:
        # 关闭连接
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def get_patient_records(patient_id):
    """
    根据患者ID查询所有诊断记录，以及相关的影像信息
    
    参数:
    patient_id (int): 患者ID
    
    返回:
    dict: 包含患者信息和诊断记录列表的结构化数据
    """
    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host="113.44.61.230",
            user="root",
            password="Ytb@210330!",
            database="medical_db"
        )
        cursor = conn.cursor(dictionary=True)  # 使用字典游标获取结果
        
        # 查询患者信息
        patient_query = """
        SELECT patient_id, patient_name, patient_gender, patient_age
        FROM patient_info
        WHERE patient_id = %s
        """
        cursor.execute(patient_query, (patient_id,))
        patient_info = cursor.fetchone()
        
        if not patient_info:
            return {"error": f"患者ID {patient_id} 不存在"}
        
        # 查询该患者的所有诊断记录
        records_query = """
        SELECT r.record_id, r.patient_id, r.fund_id, r.result, r.suggestion, r.diagnosis_date, r.user_id
        FROM record_info r
        WHERE r.patient_id = %s
        ORDER BY r.diagnosis_date DESC
        """
        cursor.execute(records_query, (patient_id,))
        records = cursor.fetchall()
        
        # 查询每条记录关联的影像信息
        for record in records:
            fund_id = record['fund_id']
            
            # 查询影像信息（包含二进制数据）
            if fund_id is not None:
                fund_query = """
                SELECT fund_id, left_fund_keyword, right_fund_keyword, patient_id, left_fund, right_fund
                FROM fund_info
                WHERE fund_id = %s
                """
                cursor.execute(fund_query, (fund_id,))
                fund_info = cursor.fetchone()
                
                # 如果存在二进制数据，将其转换为base64编码字符串
                if fund_info:
                    if fund_info['left_fund'] is not None:
                        fund_info['left_fund'] = base64.b64encode(fund_info['left_fund']).decode('utf-8')
                    if fund_info['right_fund'] is not None:
                        fund_info['right_fund'] = base64.b64encode(fund_info['right_fund']).decode('utf-8')
                
                record['fund_info'] = fund_info
            else:
                record['fund_info'] = None
        
        # 构建结构化数据
        result = {
            'patient': patient_info,
            'records': records
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

def get_fund_info(fund_id):
    """
    根据fund_id查询眼底影像信息
    
    参数:
    fund_id (int): 眼底影像ID
    
    返回:
    dict: 包含眼底影像信息的结构化数据，如果找不到则返回错误信息
    """
    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host="113.44.61.230",
            user="root",
            password="Ytb@210330!",
            database="medical_db"
        )
        cursor = conn.cursor(dictionary=True)  # 使用字典游标获取结果
        
        # 查询影像信息
        fund_query = """
        SELECT fund_id, left_fund_keyword, right_fund_keyword, patient_id, left_fund, right_fund
        FROM fund_info
        WHERE fund_id = %s
        """
        cursor.execute(fund_query, (fund_id,))
        fund_info = cursor.fetchone()
        
        if not fund_info:
            return {"error": f"眼底影像ID {fund_id} 不存在"}
        
        # 如果存在二进制数据，将其转换为base64编码字符串
        if fund_info['left_fund'] is not None:
            fund_info['left_fund'] = base64.b64encode(fund_info['left_fund']).decode('utf-8')
        if fund_info['right_fund'] is not None:
            fund_info['right_fund'] = base64.b64encode(fund_info['right_fund']).decode('utf-8')
        
        # 查询关联的患者信息
        patient_id = fund_info['patient_id']
        patient_query = """
        SELECT patient_id, patient_name, patient_gender, patient_age
        FROM patient_info
        WHERE patient_id = %s
        """
        cursor.execute(patient_query, (patient_id,))
        patient_info = cursor.fetchone()
        
        # 查询关联的诊断记录
        records_query = """
        SELECT r.record_id, r.patient_id, r.fund_id, r.result, r.suggestion, r.diagnosis_date, r.user_id
        FROM record_info r
        WHERE r.fund_id = %s
        ORDER BY r.diagnosis_date DESC
        """
        cursor.execute(records_query, (fund_id,))
        records = cursor.fetchall()
        
        # 构建结构化数据
        result = {
            'fund': fund_info,
            'patient': patient_info,
            'records': records
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

if __name__ == "__main__":
    # make_user_info(user_id=1,
    #                user_account="123456",
    #                user_password="123456",
    #                user_email="EMAIL",
    #                user_phone="1234567890",
    #                user_gender="Male",
    #                user_age=20)
    # left_fund = read_image_file("F:\BFPC\cropped_#Training_Dataset/1_left.jpg")
    # right_fund = read_image_file("F:\BFPC\cropped_#Training_Dataset/1_right.jpg")
    # save_results(patient_id=1,predict_result="糖尿病，青光眼",advise="不要吃什么",patient_age=20,patient_sex='Male',
    #              left_fund_keyword="left",
    #              right_fund_keyword="right",
    #              left_fund=left_fund,
    #              right_fund=right_fund
    #              )
    
    #示例：查询最近5条诊断记录
    # recent_records = get_recent_records(5)
    # print(f"查询到 {len(recent_records)} 条最近诊断记录")
    # for record in recent_records:
    #     print(f"记录ID: {record['record']['record_id']}, 患者: {record['patient']['patient_name']}, 诊断结果: {record['record']['result']}")
    
    # 示例：根据患者ID查询诊断记录
    # patient_records = get_patient_records(1)
    # if "error" in patient_records:
    #     print(patient_records["error"])
    # else:
    #     patient = patient_records["patient"]
    #     records = patient_records["records"]
    #     print(f"患者: {patient['patient_id']}, 共有 {len(records)} 条诊断记录")
    #     for record in records:
    #         print(f"记录ID: {record['record_id']}, 诊断日期: {record['diagnosis_date']}, 诊断结果: {record['result']}")
    
    # 示例：根据影像ID查询眼底影像信息
    # fund_info = get_fund_info(1)
    # if "error" in fund_info:
    #     print(fund_info["error"])
    # else:
    #     fund = fund_info["fund"]
    #     patient = fund_info["patient"]
    #     records = fund_info["records"]
    #     print(f"影像ID: {fund['fund_id']}, 患者: {patient['patient_name']}, 左眼关键词: {fund['left_fund_keyword']}, 右眼关键词: {fund['right_fund_keyword']}")
    #     print(f"关联诊断记录数: {len(records)}")
    #     if len(records) > 0:
    #         for record in records:
    #             print(f"记录ID: {record['record_id']}, 诊断日期: {record['diagnosis_date']}, 诊断结果: {record['result']}")
    data = get_fund_info(33)
    print(data['records'])
    
