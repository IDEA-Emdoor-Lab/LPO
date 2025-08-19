import json
import os
import re

from volcenginesdkarkruntime import Ark
import json
from tqdm import tqdm

# DeepSeek 模型 API 配置
DEEPSEEK_API_KEY = '88358db4-afa1-4a1c-8d51-10e41e770562'
DEEPSEEK_MODEL = "ep-20250407140322-x6754" # "ep-20250317104024-f6kg8" # Deep Seek R1 Chat

class DeepSeekAPI:
    def __init__(self):
        # doubao client
        self.client = Ark(api_key=DEEPSEEK_API_KEY)
        # 通用模型 20241215
        self.GENERAL_MODEL_ENDPOINT_ID = DEEPSEEK_MODEL
        self.temperature = 1
        self.top_p = 0.7
        self.frequency_penalty = 0

        self.cls_prompt_template = "你是人工智能助手"
        self.prompte_template = """给你一个任务，你需要做的是根据问题判断模型推理结果是否和gold answer相等，如果相等，直接输出是，不相等输出否
问题：{question}

模型推理结果：{answer}

gold answer：{gold_answer}"""

    def generate(self, messages):
        try:
            messages.insert(0, {'role': 'system', 'content': self.cls_prompt_template})
            resp = self.client.chat.completions.create(
                model=self.GENERAL_MODEL_ENDPOINT_ID,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                messages=messages,
            )
        except Exception as e:
            print(f"DOUBAO EXCEPTION: {str(e)}", flush=True)
            return ''
        print(f"DS | {resp.choices[0].message.content}", flush=True)
        return resp.choices[0].message.content.strip()
    
    
    def calculate_yes_ratio(self, data_list):
        yes_count = 0
        total = len(data_list)
        
        for item in data_list:
            if item.get('model_predict') == '是':
                yes_count += 1
        
        if total == 0:
            return 0.0  # 避免除以零的情况
        else:
            return yes_count / total
    
    def decode_answer(self, text):
        # 使用正则表达式提取答案
        pattern = r'#### (-?\d+)'
        match = re.search(pattern, text)
        #print(f'text is:{text}', flush=True)
        if match:
            answer = match.group(1)
            #print(f'answer is:{answer}', flush=True)
            return str(answer)
        else:
            return None
    
    def format_number_string(self, s):
        """
        判断字符串是否全为数字，如果是则按照每3个字符加逗号的方式格式化
        
        Args:
            s (str): 输入字符串
            
        Returns:
            str: 格式化后的字符串或原字符串(如果不是全数字)
        """
        # 判断是否全为数字
        if not s.isdigit():
            return s
        
        # 反转字符串便于从右向左每3位加逗号
        reversed_str = s[::-1]
        formatted = []
        
        # 每3个字符加一个逗号
        for i in range(0, len(reversed_str), 3):
            chunk = reversed_str[i:i+3]
            formatted.append(chunk)
        
        # 重新组合并反转回来
        formatted_str = ','.join(formatted)[::-1]
        
        return formatted_str
    
    def process(self, input_path, save_path):
        """_summary_

        Args:
            input_path (_type_): _description_
        """

        fout = open(save_path, "w", encoding="utf-8")

        #data_list = []

        with open(input_path, 'r', encoding='utf-8') as fin:
            #contents = json.loads(f.read())
            for line in fin:
                data_dict = json.loads(line)

                better_context = data_dict['better_context']
                candidate_set = data_dict['candidate_set']
                model_input = data_dict['model_input']
                model_input = model_input.replace('<|im_start|>user\n', '').replace('', '<|im_end|><|endoftext|><|im_start|>assistant\n')
                gold_answer = self.decode_answer(better_context)

                format_gold_answer = self.format_number_string(gold_answer)

                gold_candidate = []
                reject_candidate = []
                
                uncertain_candidate = []

                detail_info_candidate = []
                for candidate in candidate_set:
                    prompt = self.prompte_template.format(question=model_input, answer=candidate, gold_answer=better_context)
                    messages = []
                    messages.append({'role': 'user', 'content': prompt})

                    '''
                    result = self.generate(messages)

                    if '是' in result and gold_answer in candidate:
                        gold_candidate.append(candidate)
                    
                    elif '否' in result and gold_answer not in candidate:
                        reject_candidate.append(candidate)
                    else:
                        uncertain_candidate.append(candidate)
                    '''
                    if data_dict['model_input'] == "<|im_start|>user\nQuestion: Mr. Grey's house was worth $100,000. He sold the house to Mr. Brown at a profit of 10%. After one year, Mr. Brown sold the house to his other friend with a 10% loss. How much was Mr. Brown's selling price?\nLet's think step by step\nAnswer:<|im_end|><|endoftext|><|im_start|>assistant\n":
                        print(f"candidate is:{candidate}", flush=True)
                        print(f"gold_answer is:{gold_answer}", flush=True)
                    if gold_answer in candidate or format_gold_answer in candidate:
                        gold_candidate.append(candidate)
                    
                    else:
                        reject_candidate.append(candidate)

                    
                    candi_info_dict = {}
                    candi_info_dict['candidate'] = candidate
                    candi_info_dict['result'] = None
                    candi_info_dict['gold_answer'] = gold_answer

                    detail_info_candidate.append(candi_info_dict)
                if len(reject_candidate) == 0:
                    continue
                data_dict['source_candidate_set'] = data_dict['candidate_set']
                data_dict['candidate_set'] = reject_candidate
                data_dict['gold_set'] = gold_candidate
                data_dict['uncertain_set'] = uncertain_candidate

                # data_list.append(res)

                fout.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
        
        #yes_ratio = self.calculate_yes_ratio(data_list)
        #print('yes ratio:', yes_ratio, flush=True)
        print('write to file')



class MATHDeepSeekAPI:
    def __init__(self):
        # doubao client
        self.client = Ark(api_key=DEEPSEEK_API_KEY)
        # 通用模型 20241215
        self.GENERAL_MODEL_ENDPOINT_ID = DEEPSEEK_MODEL
        self.temperature = 1
        self.top_p = 0.7
        self.frequency_penalty = 0

        self.cls_prompt_template = "你是人工智能助手"
        self.prompte_template = """给你一个任务，你需要做的是根据问题判断模型推理结果是否和gold answer相等，如果相等，直接输出是，不相等输出否
问题：{question}

模型推理结果：{answer}

gold answer：{gold_answer}"""

    def generate(self, messages):
        try:
            messages.insert(0, {'role': 'system', 'content': self.cls_prompt_template})
            resp = self.client.chat.completions.create(
                model=self.GENERAL_MODEL_ENDPOINT_ID,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                messages=messages,
            )
        except Exception as e:
            print(f"DOUBAO EXCEPTION: {str(e)}", flush=True)
            return ''
        print(f"DS | {resp.choices[0].message.content}", flush=True)
        return resp.choices[0].message.content.strip()
    
    
    def calculate_yes_ratio(self, data_list):
        yes_count = 0
        total = len(data_list)
        
        for item in data_list:
            if item.get('model_predict') == '是':
                yes_count += 1
        
        if total == 0:
            return 0.0  # 避免除以零的情况
        else:
            return yes_count / total
    
    def decode_answer(self, text):
        # 使用正则表达式提取答案
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        print(f'text is:{text}', flush=True)
        if match:
            answer = match.group(1)
            print(f'answer is:{answer}', flush=True)
            return answer
        else:
            return None
    
    def process(self, input_path, save_path):
        """_summary_

        Args:
            input_path (_type_): _description_
        """

        fout = open(save_path, "w", encoding="utf-8")

        #data_list = []

        with open(input_path, 'r', encoding='utf-8') as fin:
            #contents = json.loads(f.read())
            for line in fin:
                data_dict = json.loads(line)

                better_context = data_dict['better_context']
                candidate_set = data_dict['candidate_set']
                model_input = data_dict['model_input']
                model_input = model_input.replace('<|im_start|>user\n', '').replace('', '<|im_end|><|endoftext|><|im_start|>assistant\n')
                gold_answer = self.decode_answer(better_context)

                if gold_answer is None:
                    continue
                gold_candidate = []
                reject_candidate = []
                
                uncertain_candidate = []

                detail_info_candidate = []
                for candidate in candidate_set:
                    prompt = self.prompte_template.format(question=model_input, answer=candidate, gold_answer=better_context)
                    messages = []
                    messages.append({'role': 'user', 'content': prompt})

                    '''
                    result = self.generate(messages)

                    if '是' in result and gold_answer in candidate:
                        gold_candidate.append(candidate)
                    
                    elif '否' in result and gold_answer not in candidate:
                        reject_candidate.append(candidate)
                    else:
                        uncertain_candidate.append(candidate)
                    '''
                    if gold_answer in candidate:
                        gold_candidate.append(candidate)
                    
                    else:
                        reject_candidate.append(candidate)

                    
                    candi_info_dict = {}
                    candi_info_dict['candidate'] = candidate
                    candi_info_dict['result'] = None
                    candi_info_dict['gold_answer'] = gold_answer

                    detail_info_candidate.append(candi_info_dict)
                if len(reject_candidate) == 0:
                    continue
                data_dict['source_candidate_set'] = data_dict['candidate_set']
                data_dict['candidate_set'] = reject_candidate
                data_dict['gold_set'] = gold_candidate
                data_dict['uncertain_set'] = uncertain_candidate

                # data_list.append(res)

                fout.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
        
        #yes_ratio = self.calculate_yes_ratio(data_list)
        #print('yes ratio:', yes_ratio, flush=True)
        print('write to file')

def get_jsonl_files(directory):
    """获取目录下所有以 .jsonl 结尾的文件"""
    jsonl_files = []
    for file in os.listdir(directory):
        if file.endswith(".jsonl"):
            jsonl_files.append(os.path.join(directory, file))
    return jsonl_files

#'''
input_path = '/cognitive_comp/ccnl_common_data/large/sft_audio_data/source/train/math/gsm8k/output/candidate/20250620/candidate'
save_path = '/cognitive_comp/ccnl_common_data/large/sft_audio_data/source/train/math/gsm8k/output/candidate/20250620/candidate/split_chose_reject'


client = DeepSeekAPI()

for file in os.listdir(input_path):

    if file.endswith(".json"):
        print(f'start process file: {file}')
        #jsonl_files.append(os.path.join(directory, file))
        input_file_path = os.path.join(input_path, file)
        save_file_path = os.path.join(save_path, file)
        client.process(input_path=input_file_path, save_path=save_file_path)
#'''


'''
input_path = '/cognitive_comp/ccnl_common_data/large/sft_audio_data/source/train/math/MATH/output/candidate/20250620/candidate'
save_path = '/cognitive_comp/ccnl_common_data/large/sft_audio_data/source/train/math/MATH/output/candidate/20250620/candidate/split_chose_reject'


math_client = MATHDeepSeekAPI()

for file in os.listdir(input_path):

    if file.endswith(".json"):
        print(f'start process file: {file}')
        #jsonl_files.append(os.path.join(directory, file))
        input_file_path = os.path.join(input_path, file)
        save_file_path = os.path.join(save_path, file)
        math_client.process(input_path=input_file_path, save_path=save_file_path)
'''

                
