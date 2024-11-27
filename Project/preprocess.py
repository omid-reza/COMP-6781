from operator import index

import pandas as pd

def preprocess(file_path: str):
    data = pd.read_excel(file_path,index_col=False, dtype={'Question': str, 'National': str, 'GPT 4-o': str, 'Claud 3.5': str, 'Gemini 1.5': str}).reset_index()
    data['local_index'] = data.index % 9
    for cl_num in [1, 3, 11, 13, 21, 23, 31, 33]:
        for llm in ['GPT 4-o', 'Claud 3.5', 'Gemini 1.5']:
            data.at[data.index[cl_num], llm] = {'A': 1, 'B': 2, 'C': 3}.get(data.at[data.index[cl_num], llm], data.at[data.index[cl_num], llm])
    for cl_num in [8, 18, 28, 38]:
        for llm in ['GPT 4-o', 'Claud 3.5', 'Gemini 1.5']:
            tmp_nums = list(map(int, data.at[data.index[cl_num], llm].split(',')))
            data.at[data.index[cl_num], llm] = (tmp_nums[0] - 1) * 4 + tmp_nums[1]
    data = data.drop([9, 19, 29, 39])
    data["GPT 4-o"] = pd.to_numeric(data["GPT 4-o"])
    data["Claud 3.5"] = pd.to_numeric(data["Claud 3.5"])
    data["Gemini 1.5"] = pd.to_numeric(data["Gemini 1.5"])
    return data

def convert_llm_cols_to_row(data):
    return pd.melt(data, id_vars=['Question', 'National', 'local_index'], value_vars=['GPT 4-o', 'Claud 3.5', 'Gemini 1.5'], var_name='LLM Name', value_name='LLM Response')