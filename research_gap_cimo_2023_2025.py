#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
研究gap验证脚本：根据_md_file匹配cimo_results_2023_2025和cimo_validation.xlsx数据
"""

import json
import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional

def load_jsonl_file(filepath: str) -> List[Dict[Any, Any]]:
    """
    加载JSONL文件
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"解析JSON行时出错 {filepath}: {e}")
    return data

def load_all_jsonl_results(results_dir: str) -> Dict[str, Dict]:
    """
    加载所有JSONL结果文件并建立_md_file到CIO列表的映射
    """
    md_file_to_cio = {}
    
    # 获取所有JSONL文件
    jsonl_files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    
    for file_path in jsonl_files:
        print(f"正在处理文件: {file_path}")
        jsonl_data = load_jsonl_file(file_path)
        
        for record in jsonl_data:
            md_file = record.get('_md_file', '')
            cio_list = record.get('cio_list', [])
            
            if md_file:
                # 存储完整的记录信息，不仅仅是cio_list
                md_file_to_cio[md_file] = {
                    'cio_list': cio_list,
                    'title': record.get('title', ''),
                    'authors': record.get('authors', ''),
                    'full_record': record
                }
    
    return md_file_to_cio

def safe_load_excel(file_path: str) -> Optional[pd.DataFrame]:
    """
    安全地加载Excel文件
    """
    try:
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"使用默认引擎加载Excel失败: {e}")
            df = pd.read_excel(file_path, engine='openpyxl')
        return df
    except ImportError as e:
        print(f"缺少必要的库 (pandas 或 openpyxl)，无法加载Excel文件: {e}")
        print("请运行以下命令安装所需库:")
        print("pip install pandas openpyxl")
        return None
    except Exception as e:
        print(f"加载Excel文件时出错: {e}")
        return None

def match_papers_by_md_file(validation_df: pd.DataFrame, md_file_mapping: Dict[str, Dict]) -> pd.DataFrame:
    """
    根据_md_file匹配验证数据和CIMO结果
    """
    # 创建一个新的DataFrame副本以避免修改原始数据
    result_df = validation_df.copy()
    
    # 添加新的cio_list列，初始化为空列表
    result_df['cio_list'] = None
    
    # 检查addressing_papers列是否存在
    if 'addressing_papers' not in validation_df.columns:
        print(f"警告: Excel文件中没有找到 'addressing_papers' 列")
        print(f"可用的列: {list(validation_df.columns)}")
        return result_df
    
    # 遍历验证数据的每一行
    for idx, row in validation_df.iterrows():
        addressing_papers = row['addressing_papers']
        
        # 如果addressing_papers是字符串，则处理
        if pd.notna(addressing_papers) and isinstance(addressing_papers, str):
            try:
                # 解析JSON字符串
                papers_list = json.loads(addressing_papers)
                
                # 如果是空列表，跳过
                if not papers_list:
                    continue
                
                # 遍历每个paper，尝试匹配
                matched_cio_lists = []
                for paper in papers_list:
                    paper_id = paper.get('paper_id', '')
                    if paper_id:
                        # 构造md文件名
                        md_file_candidate = paper_id + '.md'
                        
                        # 在映射中查找
                        if md_file_candidate in md_file_mapping:
                            # 匹配成功，添加CIO列表
                            cio_list = md_file_mapping[md_file_candidate]['cio_list']
                            matched_cio_lists.append({
                                'paper_id': paper_id,
                                'cio_list': cio_list
                            })
                            print(f"匹配成功: {paper_id} -> {md_file_candidate}")
                
                # 将匹配到的cio_list列表赋值给新列
                if matched_cio_lists:
                    result_df.at[idx, 'cio_list'] = str(matched_cio_lists)
                else:
                    print(f"未找到匹配: {addressing_papers[:100]}...")
                    
            except json.JSONDecodeError as e:
                print(f"解析JSON失败 (行 {idx}): {e}")
    
    return result_df

def main():
    """
    主函数
    """
    # 定义文件路径
    cimo_results_dir = "/data_share_from_3090/wy_code/code/EE/NER/utd24/cimo_results_2023_2025"
    validation_file = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_validation/cimo_validation.xlsx"
    
    print("开始加载CIMO结果数据...")
    md_file_mapping = load_all_jsonl_results(cimo_results_dir)
    print(f"建立了 {len(md_file_mapping)} 个 _md_file 的映射关系")
    
    print("\n开始加载验证数据...")
    validation_df = safe_load_excel(validation_file)
    if validation_df is None:
        print("无法加载验证数据，程序退出")
        return None, None
        
    print(f"验证数据包含 {len(validation_df)} 行记录")
    print(f"验证数据列: {list(validation_df.columns)}")
    
    print("\n开始匹配CIMO结果与验证数据...")
    matched_df = match_papers_by_md_file(validation_df, md_file_mapping)
    
    # 统计匹配情况
    matched_count = matched_df['cio_list'].notna().sum()
    total_count = len(matched_df)
    match_rate = matched_count / total_count * 100 if total_count > 0 else 0
    
    print(f"\n匹配统计:")
    print(f"总记录数: {total_count}")
    print(f"匹配成功的记录数: {matched_count}")
    print(f"匹配率: {match_rate:.2f}%")
    
    # 保存匹配结果到原文件路径（保持原格式）
    output_file = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_validation/cimo_validation.xlsx"
    try:
        matched_df.to_excel(output_file, index=False)
        print(f"\n✅ 结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
        # 尝试保存为CSV作为备选
        csv_output = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_validation/cimo_validation_2023_2025.csv"
        matched_df.to_csv(csv_output, index=False)
        print(f"结果已保存为CSV格式: {csv_output}")
    
    return matched_df, {'total': total_count, 'matched': matched_count, 'match_rate': match_rate}

if __name__ == "__main__":
    df, stats = main()
    if df is not None:
        print("\n✅ 分析完成!")
    else:
        print("\n❌ 分析失败!")