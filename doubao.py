import os
from openai import OpenAI
import re
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key
    api_key="8ef0301d-d21f-4d06-a9a0-0c00ed164c52",
)

prompt = "请根据患者的性别、年龄、病史和我们模型预测出来的疾病，给出相应的诊断报告书，\
    包括：病情、建议用药、建议检查项目、建议治疗方案等内容。输出格式化的文档，如\
        患者名称：xxx\
        患者年龄：xxx岁\
        患者性别：xxx\
        病情：xxxxxxx\
        建议用药：xxxxxxx\
        建议检查项目：xxxxxxx\
        注意事项：xxxxxxx，请严格按照格式输出，即使患者没有明确患任何疾病。"

def get_book(patient_name,patient_age,patient_sex,patient_disease,patient_history="无"):

    completion = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="doubao-1-5-lite-32k-250115",
    messages=[
        {"role": "system", "content": "你是一个眼科疾病诊断专家"},
        {"role": "user", "content": f"{prompt}患者详情：{patient_name}，{patient_age}岁，{patient_sex}，{patient_disease},{patient_history}"},

    ],
    )
    return completion.choices[0].message.content

def clean_and_format_text(text: str) -> str:
    # 去除 Markdown 风格的特殊符号
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 去除加粗 ** **
    text = re.sub(r'#*\s*(.*?)\n', r'\1\n', text)  # 去除标题符号 #
    text = re.sub(r'-\s+', '', text)  # 去除列表项的 -
    text = re.sub(r'\*+', '', text)  # 去除所有 * 号
    
    # 规范化空行，确保段落清晰
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    
    return text

if __name__ == '__main__':
    print(
        clean_and_format_text(
            get_book(
                patient_name="张三",
                patient_age="20",
                patient_sex="男",
                patient_disease="青光眼",
                patient_history="无"
            ) 
        )
    )
import os
from openai import OpenAI
import re
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key
    api_key="8ef0301d-d21f-4d06-a9a0-0c00ed164c52",
)

prompt = "请根据患者的性别、年龄、病史和我们模型预测出来的疾病，给出相应的诊断报告书，\
    包括：病情、建议用药、建议检查项目、建议治疗方案等内容。输出格式化的文档，如\
        患者名称：xxx\
        患者年龄：xxx岁\
        患者性别：xxx\
        病情：xxxxxxx\
        建议用药：xxxxxxx\
        建议检查项目：xxxxxxx\
        注意事项：xxxxxxx"

def get_book(patient_name,patient_age,patient_sex,patient_disease,patient_history="无"):

    completion = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="doubao-1-5-lite-32k-250115",
    messages=[
        {"role": "system", "content": "你是一个眼科疾病诊断专家"},
        {"role": "user", "content": f"{prompt}患者详情：{patient_name}，{patient_age}岁，{patient_sex}，{patient_disease},{patient_history}"},

    ],
    )
    return completion.choices[0].message.content

def clean_and_format_text(text: str) -> str:
    # 去除 Markdown 风格的特殊符号
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 去除加粗 ** **
    text = re.sub(r'#*\s*(.*?)\n', r'\1\n', text)  # 去除标题符号 #
    text = re.sub(r'-\s+', '', text)  # 去除列表项的 -
    text = re.sub(r'\*+', '', text)  # 去除所有 * 号
    
    # 规范化空行，确保段落清晰
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    
    return text

if __name__ == '__main__':
    print(
        clean_and_format_text(
            get_book(
                patient_name="张三",
                patient_age="20",
                patient_sex="男",
                patient_disease="青光眼",
                patient_history="无"
            ) 
        )
    )