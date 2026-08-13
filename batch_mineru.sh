# #!/bin/bash

# # 1. 设置模型下载源 (根据图片中的 Tip 建议，国内用户建议切到 modelscope)
# # 如果你的网络访问 huggingface 有困难，请务必保留这一行
# export MINERU_MODEL_SOURCE=modelscope

# # 2. 定义路径变量
# INPUT_PATH="/data_share_from_3090/wy_code/code/EE/NER/utd24/download/Golden_Paper"
# OUTPUT_PATH="/data_share_from_3090/wy_code/code/EE/NER/utd24/output/Golden_Paper"

# # 3. 检查输入路径是否存在
# if [ ! -e "$INPUT_PATH" ]; then
#     echo "错误: 输入路径不存在 -> $INPUT_PATH"
#     exit 1
# fi

# # 4. 创建输出目录（如果不存在）
# mkdir -p "$OUTPUT_PATH"

# # 5. 执行 MinerU 命令
# echo "正在开始处理文档..."
# mineru -p "$INPUT_PATH" -o "$OUTPUT_PATH"

# # 6. 检查执行结果
# if [ $? -eq 0 ]; then
#     echo "处理成功！输出已保存至: $OUTPUT_PATH"
# else
#     echo "处理过程中出现错误，请检查日志。"
# fi




# #!/bin/bash

# # 1. 设置模型下载源（国内用户建议使用 modelscope）
# export MINERU_MODEL_SOURCE=modelscope
# export MODELSCOPE_CACHE=/home/irlab/.cache/modelscope
# mineru-models-download

# # 2. 定义基础路径
# BASE_INPUT="/data_share_from_3090/wy_code/code/EE/NER/utd24/download/EBM/Industrial_Marketing_Management"
# BASE_OUTPUT="/data_share_from_3090/wy_code/code/EE/NER/utd24/output/EBM/Industrial_Marketing_Management"

# # 3. 检查基础输入路径是否存在
# if [ ! -d "$BASE_INPUT" ]; then
#     echo "错误: 基础输入路径不存在 -> $BASE_INPUT"
#     exit 1
# fi

# # 4. 循环处理 2020-2025 年
# for YEAR in {2020..2025}; do
#     INPUT_PATH="$BASE_INPUT/$YEAR"
#     OUTPUT_PATH="$BASE_OUTPUT/$YEAR"

#     echo "========================================"
#     echo "正在处理年份: $YEAR"
#     echo "输入路径: $INPUT_PATH"
#     echo "输出路径: $OUTPUT_PATH"
#     echo "========================================"

#     # 检查该年份目录是否存在
#     if [ ! -d "$INPUT_PATH" ]; then
#         echo "警告: $YEAR 年份目录不存在，跳过 -> $INPUT_PATH"
#         echo ""
#         continue
#     fi

#     # 检查该目录下是否有 PDF 文件
#     PDF_COUNT=$(find "$INPUT_PATH" -maxdepth 1 -name "*.pdf" | wc -l)
#     if [ "$PDF_COUNT" -eq 0 ]; then
#         echo "警告: $YEAR 目录下未找到 PDF 文件，跳过。"
#         echo ""
#         continue
#     fi

#     echo "发现 $PDF_COUNT 个 PDF 文件，开始处理..."

#     # 创建输出目录
#     mkdir -p "$OUTPUT_PATH"

#     # 执行 MinerU
#     mineru -p "$INPUT_PATH" -o "$OUTPUT_PATH"

#     # 检查执行结果
#     if [ $? -eq 0 ]; then
#         echo "✅ $YEAR 处理成功！输出已保存至: $OUTPUT_PATH"
#     else
#         echo "❌ $YEAR 处理失败，请检查日志。"
#     fi

#     echo ""
# done

# echo "========================================"
# echo "全部年份处理完毕！"
# echo "========================================"




# CUDA_VISIBLE_DEVICES=2 MINERU_MODEL_SOURCE=modelscope \mineru -p /data_share_from_3090/wy_code/code/EE/NER/utd24/download/EBM/Journal_of_Supply_Chain_Management/2025/jscm.12335.pdf 
#      -o /data_share_from_3090/wy_code/code/EE/NER/utd24/output/EBM/Journal_of_Supply_Chain_Management/2025/jscm.12335

#  CUDA_VISIBLE_DEVICES=2 MINERU_MODEL_SOURCE=modelscope



#!/bin/bash

# 1. 设置模型下载源
export MINERU_MODEL_SOURCE=modelscope
export MODELSCOPE_CACHE=/home/irlab/.cache/modelscope
mineru-models-download



# 2. 定义基础路径
BASE_INPUT="/data_share_from_3090/wy_code/code/EE/NER/utd24/download/EBM/Strategic_Management_Journal"
BASE_OUTPUT="/data_share_from_3090/wy_code/code/EE/NER/utd24/output/EBM/Strategic_Management_Journal"

# 3. 检查基础输入路径是否存在
if [ ! -d "$BASE_INPUT" ]; then
    echo "错误: 基础输入路径不存在 -> $BASE_INPUT"
    exit 1
fi

FAILED_LIST=()

# 4. 循环处理 2020-2025 年
for YEAR in {2020..2025}; do
    INPUT_PATH="$BASE_INPUT/$YEAR"
    OUTPUT_PATH="$BASE_OUTPUT/$YEAR"

    echo "========================================"
    echo "正在处理年份: $YEAR"
    echo "========================================"

    if [ ! -d "$INPUT_PATH" ]; then
        echo "警告: $YEAR 年份目录不存在，跳过 -> $INPUT_PATH"
        continue
    fi

    mkdir -p "$OUTPUT_PATH"

    # 5. 逐个处理 PDF，检查是否已有输出
    while IFS= read -r -d '' PDF_FILE; do
        BASENAME=$(basename "$PDF_FILE" .pdf)

        # MinerU 默认输出结构: <OUTPUT_PATH>/<pdf文件名>/auto/<pdf文件名>.md
        EXPECTED_OUTPUT="$OUTPUT_PATH/$BASENAME/hybrid_auto/$BASENAME.md"

        if [ -f "$EXPECTED_OUTPUT" ]; then
            echo "⏭️  已存在，跳过: $BASENAME"
            continue
        fi

        echo "🔄 正在转换: $BASENAME"

        # 单文件转换（-p 支持单个文件路径）
        mineru -p "$PDF_FILE" -o "$OUTPUT_PATH"

        if [ $? -eq 0 ] && [ -f "$EXPECTED_OUTPUT" ]; then
            echo "✅ 转换成功: $BASENAME"
        else
            echo "❌ 转换失败: $BASENAME"
            FAILED_LIST+=("$PDF_FILE")
        fi

    done < <(find "$INPUT_PATH" -maxdepth 1 -name "*.pdf" -print0)

    echo ""
done
# 6. 汇总失败列表
echo "========================================"
if [ ${#FAILED_LIST[@]} -eq 0 ]; then
    echo "🎉 全部 PDF 转换成功！"
else
    echo "❌ 以下 ${#FAILED_LIST[@]} 个 PDF 转换失败："
    for F in "${FAILED_LIST[@]}"; do
        echo "   - $F"
    done
    echo ""
    echo "可将失败列表写入文件，便于后续排查："
    printf '%s\n' "${FAILED_LIST[@]}" > failed_pdfs.txt
    echo "已保存至: failed_pdfs.txt"
fi
echo "========================================"
