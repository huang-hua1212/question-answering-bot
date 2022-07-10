# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:06:02 2020

@author: User
"""

import openpyxl
 


'''操作xlsx格式的表格文件'''
def write_excel_xlsx(path, sheet_name, value):
    index = len(value)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.cell(row=i+1, column=j+1, value=value[i][j])
    workbook.save(path)
    
write_excel_xlsx("test.xlsx","test_table",
                 [["Question","q_movestopwords","get_inverted_idx","get_total_q",
                  "get_preprocess","get_tokenized_text","get_tokens_segments_tensor",
                  "get_encoded_layers","get_sentence_embedding","get_new_bert",
                  "get_cos_list","get_top_index","get_answer"]])
    
