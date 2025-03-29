import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm, mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image, Table, TableStyle, Frame, PageTemplate, BaseDocTemplate, NextPageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
from PIL import Image as PILImage
import base64
from datetime import datetime
import tempfile
from predict import parse_medical_record
# 尝试注册中文字体
font_registered = False
default_font_name = 'Helvetica'

# 尝试多种可能的字体路径
possible_fonts = [
    ('SimSun', 'SimSun.ttf'),  # 当前目录下的字体
    ('SimSun', os.path.join(os.path.dirname(__file__), 'SimSun.ttf')),  # 脚本目录下的字体
    ('SimSun', 'C:\\Windows\\Fonts\\simsun.ttc'),  # Windows系统字体
    ('SimSun', 'C:\\Windows\\Fonts\\simfang.ttf'),  # Windows系统字体备选
    ('MSGothic', 'C:\\Windows\\Fonts\\msgothic.ttc'),  # 日语字体，也支持中文
    ('Microsoft YaHei', 'C:\\Windows\\Fonts\\msyh.ttc'),  # 微软雅黑字体
]

for font_name, font_path in possible_fonts:
    try:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            default_font_name = font_name
            font_registered = True
            print(f"成功注册字体: {font_name} 来自 {font_path}")
            break
    except Exception as e:
        print(f"注册字体 {font_name} 失败: {e}")

if not font_registered:
    print("警告: 无法注册任何中文字体，将使用默认字体。文档中的中文可能无法正确显示。")

# 定义样式
def get_styles():
    styles = getSampleStyleSheet()
    
    # 标题样式
    styles.add(ParagraphStyle(
        name='ChineseTitle',
        fontName=default_font_name,
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=0.5*cm,
        textColor=colors.darkblue,
        leading=22
    ))
    
    # 副标题样式
    styles.add(ParagraphStyle(
        name='ChineseHeading',
        fontName=default_font_name,
        fontSize=14,
        alignment=TA_LEFT,
        spaceAfter=0.2*cm,
        textColor=colors.darkblue,
        leading=16,
        borderWidth=1,
        borderColor=colors.lightgrey,
        borderPadding=5,
        borderRadius=2,
        backColor=colors.lightgrey.clone(alpha=0.2)
    ))
    
    # 普通文本样式
    styles.add(ParagraphStyle(
        name='ChineseNormal',
        fontName=default_font_name,
        fontSize=11,
        alignment=TA_LEFT,
        spaceAfter=0.1*cm,
        leading=14
    ))
    
    # 列表项样式
    styles.add(ParagraphStyle(
        name='ChineseList',
        fontName=default_font_name,
        fontSize=11,
        alignment=TA_LEFT,
        leftIndent=0.5*cm,
        leading=14,
        bulletIndent=0.2*cm
    ))
    
    # 页眉样式
    styles.add(ParagraphStyle(
        name='Header',
        fontName=default_font_name,
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    ))
    
    # 页脚样式
    styles.add(ParagraphStyle(
        name='Footer',
        fontName=default_font_name,
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    ))
    
    # 表格标题样式
    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName=default_font_name,
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.white,
        leading=12
    ))
    
    return styles

# 创建页眉和页脚
def header_footer(canvas, doc, hospital_name, hospital_logo_path=None):
    canvas.saveState()
    width, height = A4
    
    # 设置页眉
    canvas.setFont(default_font_name, 8)
    canvas.setFillColor(colors.grey)
    
    # 添加医院logo（如果有）
    if hospital_logo_path and os.path.exists(hospital_logo_path):
        try:
            canvas.drawImage(hospital_logo_path, 30, height - 40, width=50, height=30, preserveAspectRatio=True)
        except:
            pass
    
    # 医院名称
    canvas.drawString(100, height - 30, hospital_name)
    
    # 在页眉下方添加分隔线
    canvas.setStrokeColor(colors.lightblue)
    canvas.line(30, height - 45, width - 30, height - 45)
    
    # 设置页脚
    canvas.setFont(default_font_name, 8)
    
    # 添加页码
    page_num = canvas.getPageNumber()
    text = f"第 {page_num} 页"
    canvas.drawString(width/2 - 20, 30, text)
    
    # 添加日期
    current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    canvas.drawString(width - 120, 30, f"生成日期: {current_date}")
    
    # 在页脚上方添加分隔线
    canvas.setStrokeColor(colors.lightblue)
    canvas.line(30, 40, width - 30, 40)
    
    canvas.restoreState()

def create_diagnostic_pdf(data, output_path=None, hospital_name="眼科诊断中心", hospital_logo_path=None):
    """
    生成眼科诊断书PDF

    参数:
    data (dict): 包含以下字段的字典:
        - patient_id (str): 患者ID
        - patient_name (str): 患者姓名
        - patient_age (str): 患者年龄
        - patient_gender (str): 患者性别
        - left_eye_keywords (str, 可选): 左眼诊断关键词
        - right_eye_keywords (str, 可选): 右眼诊断关键词
        - left_eye_image (str, 可选): 左眼图片的Base64编码
        - right_eye_image (str, 可选): 右眼图片的Base64编码
        - diagnosis (str): 诊断所患疾病描述
        - medication (str, 可选): 建议用药
        - examinations (str, 可选): 建议检查项目
        - precautions (list, 可选): 注意事项列表
        - doctor_name (str): 诊断医生姓名
    output_path (str, 可选): PDF输出路径。如果为None，则返回PDF数据的BytesIO对象
    hospital_name (str, 可选): 医院或诊断中心名称
    hospital_logo_path (str, 可选): 医院logo文件路径

    返回:
    如果output_path为None，返回BytesIO对象；否则返回None
    """
    # 检查必要的字段
    required_fields = ['patient_id', 'patient_name', 'patient_age', 'patient_gender', 
                       'diagnosis', 'doctor_name']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValueError(f'缺少必要字段: {", ".join(missing_fields)}')
    
    # 创建临时文件列表用于后续清理
    temp_files = []
    
    # 创建PDF缓冲区
    buffer = io.BytesIO()
    
    # 创建一个自定义的文档模板，带有页眉页脚
    class DiagnosticDocTemplate(BaseDocTemplate):
        def __init__(self, filename, **kw):
            BaseDocTemplate.__init__(self, filename, **kw)
            template = PageTemplate('normal', [
                Frame(
                    self.leftMargin, 
                    self.bottomMargin, 
                    self.width, 
                    self.height - 1.5*cm, 
                    id='normal'
                )], 
                onPage=lambda canvas, doc: header_footer(canvas, doc, hospital_name, hospital_logo_path)
            )
            self.addPageTemplates([template])
    
    # 使用自定义文档模板
    doc = DiagnosticDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=3*cm,  # 增加顶部边距，为页眉留出空间
        bottomMargin=2*cm
    )
    
    styles = get_styles()
    elements = []
    
    # 添加装饰性元素 - 顶部色带
    elements.append(Spacer(1, 0.2*cm))
    
    # 标题带有下划线
    title = Paragraph(f"<u>眼科诊断书</u>", styles['ChineseTitle'])
    elements.append(title)
    elements.append(Spacer(1, 0.5*cm))
    
    # 创建患者基本信息表格
    patient_data = [
        [Paragraph("患者ID", styles['TableHeader']), Paragraph(data.get('patient_id', ''), styles['ChineseNormal']),
         Paragraph("患者姓名", styles['TableHeader']), Paragraph(data.get('patient_name', ''), styles['ChineseNormal'])],
        [Paragraph("患者年龄", styles['TableHeader']), Paragraph(f"{data.get('patient_age', '')}岁", styles['ChineseNormal']),
         Paragraph("患者性别", styles['TableHeader']), Paragraph(data.get('patient_gender', ''), styles['ChineseNormal'])]
    ]
    
    patient_table = Table(patient_data, colWidths=[3*cm, 4*cm, 3*cm, 4*cm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    elements.append(Paragraph("患者基本信息", styles['ChineseHeading']))
    elements.append(Spacer(1, 0.2*cm))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.5*cm))
    
    # 诊断关键词
    elements.append(Paragraph("诊断关键词", styles['ChineseHeading']))
    
    # 创建诊断关键词表格
    keywords_data = [
        [Paragraph("左眼诊断关键词", styles['TableHeader']), Paragraph(data.get('left_eye_keywords', ''), styles['ChineseNormal'])],
        [Paragraph("右眼诊断关键词", styles['TableHeader']), Paragraph(data.get('right_eye_keywords', ''), styles['ChineseNormal'])]
    ]
    
    keywords_table = Table(keywords_data, colWidths=[4*cm, 10*cm])
    keywords_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    elements.append(Spacer(1, 0.2*cm))
    elements.append(keywords_table)
    elements.append(Spacer(1, 0.5*cm))
    
    # 眼底图片
    elements.append(Paragraph("眼底图片", styles['ChineseHeading']))
    elements.append(Spacer(1, 0.2*cm))
    
    # 创建包含双眼图片的表格
    eye_images = []
    left_eye_image = None
    right_eye_image = None
    
    # 处理左眼图片
    if 'left_eye_image' in data and data['left_eye_image']:
        try:
            # 使用PIL直接处理图像数据
            img_data = base64.b64decode(data['left_eye_image'])
            img_buffer = io.BytesIO(img_data)
            pil_img = PILImage.open(img_buffer)
            
            # 保存到临时文件
            tmp_left_path = os.path.join(os.path.dirname(__file__), f"tmp_left_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
            pil_img.save(tmp_left_path)
            temp_files.append(tmp_left_path)
            
            # 创建图片对象
            left_eye_image = Image(tmp_left_path, width=7*cm, height=7*cm)
            
        except Exception as e:
            left_eye_image = Paragraph(f"左眼图片加载失败: {str(e)}", styles['ChineseNormal'])
    
    # 处理右眼图片
    if 'right_eye_image' in data and data['right_eye_image']:
        try:
            # 使用PIL直接处理图像数据
            img_data = base64.b64decode(data['right_eye_image'])
            img_buffer = io.BytesIO(img_data)
            pil_img = PILImage.open(img_buffer)
            
            # 保存到临时文件
            tmp_right_path = os.path.join(os.path.dirname(__file__), f"tmp_right_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
            pil_img.save(tmp_right_path)
            temp_files.append(tmp_right_path)
            
            # 创建图片对象
            right_eye_image = Image(tmp_right_path, width=7*cm, height=7*cm)
            
        except Exception as e:
            right_eye_image = Paragraph(f"右眼图片加载失败: {str(e)}", styles['ChineseNormal'])
    
    # 创建图片表格
    if left_eye_image or right_eye_image:
        # 创建表头
        eye_images.append([
            Paragraph("左眼眼底图像", styles['TableHeader']),
            Paragraph("右眼眼底图像", styles['TableHeader'])
        ])
        
        # 添加图片单元格
        eye_images.append([
            left_eye_image if left_eye_image else Paragraph("无左眼图像", styles['ChineseNormal']),
            right_eye_image if right_eye_image else Paragraph("无右眼图像", styles['ChineseNormal'])
        ])
        
        # 创建表格
        eye_table = Table(eye_images, colWidths=[7.5*cm, 7.5*cm])
        eye_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        elements.append(eye_table)
    else:
        elements.append(Paragraph("未提供眼底图像", styles['ChineseNormal']))
    
    elements.append(Spacer(1, 0.5*cm))
    
    # 诊断疾病
    elements.append(Paragraph("诊断疾病", styles['ChineseHeading']))
    elements.append(Spacer(1, 0.2*cm))
    
    # 将诊断内容放在带框的段落中
    diagnosis_style = ParagraphStyle(
        'DiagnosisStyle',
        parent=styles['ChineseNormal'],
        borderWidth=1,
        borderColor=colors.lightgrey,
        borderPadding=8,
        borderRadius=5,
        leading=14
    )
    diagnosis_para = Paragraph(f"{data.get('diagnosis', '')}", diagnosis_style)
    elements.append(diagnosis_para)
    elements.append(Spacer(1, 0.5*cm))
    
    # 诊断建议
    elements.append(Paragraph("诊断建议", styles['ChineseHeading']))
    elements.append(Spacer(1, 0.2*cm))
    
    # 用药建议
    if 'medication' in data:
        med_style = ParagraphStyle(
            'MedicationStyle',
            parent=styles['ChineseNormal'],
            borderWidth=0,
            borderColor=colors.lightblue,
            borderPadding=5,
            leftIndent=10,
            leading=14
        )
        elements.append(Paragraph("<b>建议用药：</b>", styles['ChineseNormal']))
        elements.append(Paragraph(f"{data.get('medication', '')}", med_style))
        elements.append(Spacer(1, 0.3*cm))
    
    # 检查建议
    if 'examinations' in data:
        exam_style = ParagraphStyle(
            'ExamStyle',
            parent=styles['ChineseNormal'],
            borderWidth=0,
            borderColor=colors.lightblue,
            borderPadding=5,
            leftIndent=10,
            leading=14
        )
        elements.append(Paragraph("<b>建议检查项目：</b>", styles['ChineseNormal']))
        elements.append(Paragraph(f"{data.get('examinations', '')}", exam_style))
        elements.append(Spacer(1, 0.3*cm))
    
    # 注意事项
    if 'precautions' in data and isinstance(data['precautions'], list) and data['precautions']:
        elements.append(Paragraph("<b>注意事项：</b>", styles['ChineseNormal']))
        
        # 创建带序号的注意事项表格
        precaution_data = []
        for i, item in enumerate(data['precautions']):
            precaution_data.append([
                Paragraph(f"{i+1}.", styles['ChineseNormal']),
                Paragraph(item, styles['ChineseNormal'])
            ])
        
        precaution_table = Table(precaution_data, colWidths=[1*cm, 14*cm])
        precaution_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        elements.append(precaution_table)
    
    elements.append(Spacer(1, 1*cm))
    
    # 医生签名和日期
    signature_data = [
        [Paragraph("", styles['ChineseNormal']), Paragraph("", styles['ChineseNormal']), 
         Paragraph("<b>诊断医生：</b>" + data.get('doctor_name', ''), styles['ChineseNormal'])],
        [Paragraph("", styles['ChineseNormal']), Paragraph("", styles['ChineseNormal']), 
         Paragraph(f"诊断日期：{datetime.now().strftime('%Y年%m月%d日')}", styles['ChineseNormal'])]
    ]
    
    signature_table = Table(signature_data, colWidths=[5*cm, 5*cm, 5*cm])
    signature_table.setStyle(TableStyle([
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(signature_table)
    
    try:
        # 生成PDF
        doc.build(elements)
        buffer.seek(0)
    finally:
        # 清理临时文件
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception as e:
                print(f"删除临时文件 {tmp_file} 失败: {e}")
    
    # 如果指定了输出路径，则保存到文件
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(buffer.getvalue())
        return None
    else:
        return buffer

def encode_image(image_path):
    """
    将图片文件编码为base64字符串
    
    参数:
    image_path (str): 图片文件路径
    
    返回:
    str: base64编码的图片数据
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"图片编码失败: {str(e)}")
    
def make_styles_data(dic_data):
    patient_id = dic_data['patient']['patient_id']
    patient_name = dic_data['patient']['patient_name']
    patient_age = dic_data['patient']['patient_age']
    patient_gender = dic_data['patient']['patient_gender']
    left_eye_keywords = dic_data['fund']['left_fund_keyword']
    right_eye_keywords = dic_data['fund']['right_fund_keyword']
    left_eye = dic_data['fund']['left_fund']
    right_eye = dic_data['fund']['right_fund']
    diagnosis = dic_data['records'][0]['result'][:-1]
    suggest = dic_data['records'][0]['suggestion']
    suggest_dic = parse_medical_record(suggest)
    examinations = suggest_dic['建议检查项目']
    precautions = suggest_dic['注意事项']
    medication = suggest_dic['建议用药']
    doctor_name = "李医生"

    styles_data = {
        "patient_id": patient_id,
        "patient_name": patient_name, 
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "left_eye_keywords": left_eye_keywords,
        "right_eye_keywords": right_eye_keywords,
        "left_eye_image": left_eye,
        "right_eye_image": right_eye,
        "diagnosis": diagnosis,
        "medication": medication,
        "examinations": examinations,
        "precautions": precautions,
        "doctor_name": doctor_name

    }

    return styles_data
    

# 示例用法
if __name__ == "__main__":
    # 示例数据
    sample_data = {
        "patient_id": "P12345",
        "patient_name": "张三",
        "patient_age": "20",
        "patient_gender": "男",
        "left_eye_keywords": "视网膜色素变性、黄斑区萎缩",
        "right_eye_keywords": "视网膜色素变性、黄斑区萎缩",
        "diagnosis": "青光眼、AMD",
        "medication": "根据青光眼类型及病情选择合适的降眼压药物，如β-受体阻滞剂（噻吗洛尔等）、前列腺素类衍生物（拉坦前列素等）、碳酸酐酶抑制剂（布林佐胺等）等。",
        "examinations": "眼压测量（多次测量）、视野检查、眼底检查、房角检查等。",
        "precautions": [
            "严格按照医嘱用药，不要自行增减药量或停药。",
            "避免长时间在暗环境中停留，如看电影、待在暗室等。",
            "定期复查眼压、视野、眼底等，观察病情变化。",
            "保持情绪稳定，避免情绪波动过大导致眼压升高。",
            "避免使用可能升高眼压的药物，如抗组胺药、抗震颤麻痹药等。",
            "睡眠时可适当垫高头部，有助于减轻眼部充血。",
            "如出现眼痛、视力下降、头痛等症状加重，及时就医。"
        ],
        "doctor_name": "李医生"
    }
    
    # 添加眼底图片（如果有）
    sample_left_eye_path = "F:\BFPC\cropped_#Training_Dataset/1_left.jpg"
    sample_right_eye_path = "F:\BFPC\cropped_#Training_Dataset/1_right.jpg"
    
    if os.path.exists(sample_left_eye_path):
        try:
            sample_data["left_eye_image"] = encode_image(sample_left_eye_path)
            print(f"成功加载左眼图片: {sample_left_eye_path}")
        except Exception as e:
            print(f"加载左眼图片失败: {e}")
    else:
        print(f"左眼图片文件不存在: {sample_left_eye_path}")
    
    if os.path.exists(sample_right_eye_path):
        try:
            sample_data["right_eye_image"] = encode_image(sample_right_eye_path)
            print(f"成功加载右眼图片: {sample_right_eye_path}")
        except Exception as e:
            print(f"加载右眼图片失败: {e}")
    else:
        print(f"右眼图片文件不存在: {sample_right_eye_path}")
    
    # 生成并保存PDF
    output_file = f"诊断书_{sample_data['patient_name']}_{datetime.now().strftime('%Y%m%d')}.pdf"
    try:
        # 可以提供医院名称和Logo路径（如果有）
        hospital_name = "IEDD眼科疾病智能诊断系统"
        # hospital_logo_path = "hospital_logo.png"  # 如果有Logo可以取消注释
        
        create_diagnostic_pdf(
            sample_data, 
            output_file, 
            hospital_name=hospital_name
            # hospital_logo_path=hospital_logo_path  # 如果有Logo可以取消注释
        )
        print(f"诊断书已生成：{output_file}")
    except Exception as e:
        print(f"生成诊断书时发生错误: {e}") 