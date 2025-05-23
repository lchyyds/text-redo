 # -*- coding: utf-8 -*-
"""
文本分析示例
"""

#%% 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import jieba
import jieba.analyse
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#%% 创建结果保存目录
# 创建results文件夹用于保存分析结果
if not os.path.exists('results'):
    os.makedirs('results')
    print("创建results文件夹成功")

#%% 数据加载
"""
输入: 无 (示例数据在函数内部生成)
输出: 
    - texts: 包含文本内容的列表
    - df: 包含文本数据的DataFrame，有'文本ID', '文本内容', '类别', '发布日期'等字段
"""
def load_sample_data():
    # 创建示例文本数据
    texts = [
        "人工智能技术正在快速发展，为各行各业带来新的机遇和挑战。",
        "数据分析在金融领域的应用越来越广泛，帮助投资者做出更明智的决策。",
        "机器学习算法可以从海量数据中学习规律，提高预测准确性。",
        "自然语言处理技术使计算机能够理解和生成人类语言，促进人机交互。",
        "深度学习在图像识别领域取得了突破性进展，准确率超过人类。",
        "大数据技术为企业提供了全新的商业洞察，帮助优化运营策略。",
        "云计算降低了企业的IT成本，提高了系统的可扩展性和灵活性。",
        "区块链技术具有去中心化和不可篡改的特点，在金融领域应用前景广阔。",
        "物联网将各种设备连接起来，实现智能化控制和管理。",
        "5G技术的商用将推动各行业数字化转型，创造新的商业模式。"
    ]
    
    # 创建DataFrame
    df = pd.DataFrame({
        '文本ID': range(1, len(texts) + 1),
        '文本内容': texts,
        '类别': ['技术'] * 5 + ['商业'] * 5,
        '发布日期': pd.date_range('20230101', periods=len(texts))
    })
    
    print("数据加载完成，共加载{}条文本".format(len(texts)))
    print("\n数据预览：")
    print(df.head())
    
    # 保存原始数据
    df.to_csv('results/原始文本数据.csv', index=False, encoding='utf-8-sig')
    print("原始数据已保存至 'results/原始文本数据.csv'")
    
    return texts, df

# 加载示例数据
texts, df = load_sample_data()

#%% 文本预处理
"""
输入: 
    - texts: 包含文本内容的列表
输出:
    - cleaned_texts: 清洗后的文本列表
    - segmented_texts: 分词后的文本列表，每个元素是一个词语列表
"""
def preprocess_texts(texts):
    # 文本清洗
    cleaned_texts = []
    for text in texts:
        # 去除特殊字符和标点符号
        text = re.sub(r'[^\w\s]', '', text)
        cleaned_texts.append(text)
    
    # 分词
    segmented_texts = []
    for text in cleaned_texts:
        words = jieba.lcut(text)
        # 去除停用词 (这里简化处理，实际应加载停用词表)
        words = [word for word in words if len(word) > 1]
        segmented_texts.append(words)
    
    # 保存处理结果
    with open('results/分词结果.txt', 'w', encoding='utf-8') as f:
        for i, words in enumerate(segmented_texts):
            f.write(f"文本{i+1}: {' '.join(words)}\n")
    
    print("\n文本预处理完成")
    print("示例分词结果:")
    print(segmented_texts[0])
    print("分词结果已保存至 'results/分词结果.txt'")
    
    return cleaned_texts, segmented_texts

# 预处理文本
cleaned_texts, segmented_texts = preprocess_texts(df['文本内容'].tolist())

#%% 词频统计与可视化
"""
输入:
    - segmented_texts: 分词后的文本列表
输出:
    - word_freq: 词频统计结果，Counter对象
    - 生成词频统计图表和词云图，保存在results文件夹
"""
def analyze_word_frequency(segmented_texts):
    # 合并所有词语
    all_words = []
    for words in segmented_texts:
        all_words.extend(words)
    
    # 统计词频
    word_freq = Counter(all_words)
    
    # 获取前20个高频词
    top_words = word_freq.most_common(20)
    
    # 保存词频统计结果
    with open('results/词频统计.txt', 'w', encoding='utf-8') as f:
        for word, freq in word_freq.most_common():
            f.write(f"{word}: {freq}\n")
    
    # 绘制词频柱状图
    plt.figure(figsize=(12, 6))
    words, freqs = zip(*top_words)
    plt.bar(words, freqs)
    plt.title('高频词统计')
    plt.xlabel('词语')
    plt.ylabel('频次')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/高频词统计.png', dpi=300)
    plt.show()
    
    # 生成词云
    wordcloud_text = ' '.join(all_words)
    wordcloud = WordCloud(
        font_path='simhei.ttf',  # 需要指定中文字体路径
        width=800,
        height=400,
        background_color='white'
    )
    try:
        wordcloud.generate(wordcloud_text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/词云图.png', dpi=300)
        plt.show()
        print("词云图已保存至 'results/词云图.png'")
    except Exception as e:
        print(f"生成词云图时出错: {e}")
        print("请确保已安装wordcloud库并提供了有效的中文字体路径")
    
    print("\n词频分析完成")
    print("词频统计已保存至 'results/词频统计.txt'")
    print("词频图表已保存至 'results/高频词统计.png'")
    
    return word_freq

# 分析词频
word_freq = analyze_word_frequency(segmented_texts)

#%% 关键词提取
"""
输入:
    - texts: 原始文本列表
输出:
    - keywords_tfidf: 基于TF-IDF的关键词列表
    - keywords_textrank: 基于TextRank的关键词列表
"""
def extract_keywords(texts):
    # 使用TF-IDF提取关键词
    keywords_tfidf = []
    for text in texts:
        keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=True, allowPOS=())
        keywords_tfidf.append(keywords)
    
    # 使用TextRank提取关键词
    keywords_textrank = []
    for text in texts:
        keywords = jieba.analyse.textrank(text, topK=5, withWeight=True, allowPOS=())
        keywords_textrank.append(keywords)
    
    # 保存关键词提取结果
    with open('results/关键词提取结果.txt', 'w', encoding='utf-8') as f:
        for i, (text, kw_tfidf, kw_textrank) in enumerate(zip(texts, keywords_tfidf, keywords_textrank)):
            f.write(f"文本{i+1}:\n{text}\n")
            f.write("TF-IDF关键词: " + ', '.join([f"{w}({s:.3f})" for w, s in kw_tfidf]) + "\n")
            f.write("TextRank关键词: " + ', '.join([f"{w}({s:.3f})" for w, s in kw_textrank]) + "\n\n")
    
    print("\n关键词提取完成")
    print("示例文本的TF-IDF关键词:")
    print(keywords_tfidf[0])
    print("关键词提取结果已保存至 'results/关键词提取结果.txt'")
    
    return keywords_tfidf, keywords_textrank

# 提取关键词
keywords_tfidf, keywords_textrank = extract_keywords(df['文本内容'].tolist())

#%% 文本向量化与主题建模
"""
输入:
    - texts: 原始文本列表
输出:
    - tfidf_matrix: TF-IDF向量化结果
    - lda_model: LDA主题模型
    - topics: 主题词列表
"""
def vectorize_and_topic_modeling(texts):
    # 文本向量化 (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        token_pattern=r"(?u)\b\w+\b",  # 适用于中文分词结果
        use_idf=True
    )
    
    # 将文本转换为分词后的字符串
    processed_texts = [' '.join(jieba.lcut(text)) for text in texts]
    
    # 向量化
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
    
    # 特征名称
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    print("\n文本向量化完成")
    print(f"向量维度: {tfidf_matrix.shape}")
    
    # LDA主题建模
    n_topics = 2  # 主题数量
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10
    )
    
    lda.fit(tfidf_matrix)
    
    # 获取每个主题的关键词
    n_top_words = 10
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
    
    # 保存主题模型结果
    with open('results/主题模型结果.txt', 'w', encoding='utf-8') as f:
        for i, topic_words in enumerate(topics):
            f.write(f"主题 {i+1}: {', '.join(topic_words)}\n")
    
    # 可视化主题词
    fig, axes = plt.subplots(1, n_topics, figsize=(15, 5))
    for i, (topic_words, ax) in enumerate(zip(topics, axes.flatten() if n_topics > 1 else [axes])):
        top_words_weights = [lda.components_[i][tfidf_vectorizer.vocabulary_[word]] for word in topic_words]
        ax.barh(range(len(topic_words)), top_words_weights, align='center')
        ax.set_yticks(range(len(topic_words)))
        ax.set_yticklabels(topic_words)
        ax.set_title(f'主题 {i+1}')
    
    plt.tight_layout()
    plt.savefig('results/主题模型可视化.png', dpi=300)
    plt.show()
    
    print("\n主题建模完成")
    print("主题模型结果已保存至 'results/主题模型结果.txt'")
    print("主题模型可视化已保存至 'results/主题模型可视化.png'")
    
    # 计算文档-主题分布
    doc_topic_dist = lda.transform(tfidf_matrix)
    
    # 保存文档-主题分布
    doc_topic_df = pd.DataFrame(doc_topic_dist, columns=[f'主题{i+1}' for i in range(n_topics)])
    doc_topic_df['文本ID'] = df['文本ID']
    doc_topic_df.to_csv('results/文档主题分布.csv', index=False, encoding='utf-8-sig')
    
    print("文档-主题分布已保存至 'results/文档主题分布.csv'")
    
    return tfidf_matrix, lda, topics

# 文本向量化与主题建模
tfidf_matrix, lda_model, topics = vectorize_and_topic_modeling(df['文本内容'].tolist())

#%% 文本分类分析
"""
输入:
    - df: 包含文本数据的DataFrame
输出:
    - 按类别统计的结果和可视化
"""
def analyze_by_category(df):
    # 按类别统计文本数量
    category_counts = df['类别'].value_counts()
    
    # 可视化类别分布
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar')
    plt.title('文本类别分布')
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig('results/文本类别分布.png', dpi=300)
    plt.show()
    
    # 按类别分析关键词
    category_keywords = {}
    for category in df['类别'].unique():
        category_texts = df[df['类别'] == category]['文本内容'].tolist()
        # 合并该类别的所有文本
        combined_text = ' '.join(category_texts)
        # 提取关键词
        keywords = jieba.analyse.extract_tags(combined_text, topK=10, withWeight=True)
        category_keywords[category] = keywords
    
    # 保存类别关键词结果
    with open('results/类别关键词分析.txt', 'w', encoding='utf-8') as f:
        for category, keywords in category_keywords.items():
            f.write(f"{category}类别关键词: " + ', '.join([f"{w}({s:.3f})" for w, s in keywords]) + "\n")
    
    print("\n类别分析完成")
    print("类别分布可视化已保存至 'results/文本类别分布.png'")
    print("类别关键词分析已保存至 'results/类别关键词分析.txt'")
    
    return category_counts, category_keywords

# 按类别分析
category_counts, category_keywords = analyze_by_category(df)

#%% 结果汇总与报告
"""
生成分析报告，汇总所有分析结果
"""
def generate_report():
    report_content = """# 文本分析报告

## 1. 数据概览
- 总文本数: {total_texts}
- 类别数: {total_categories}
- 类别分布: {category_distribution}

## 2. 文本预处理
- 已完成文本清洗和分词
- 分词结果保存在 'results/分词结果.txt'

## 3. 词频分析
- 已生成高频词统计和词云图
- 结果保存在 'results/词频统计.txt' 和 'results/高频词统计.png'

## 4. 关键词提取
- 使用TF-IDF和TextRank算法提取关键词
- 结果保存在 'results/关键词提取结果.txt'

## 5. 主题建模
- 使用LDA算法进行主题建模，共提取{n_topics}个主题
- 主题词列表保存在 'results/主题模型结果.txt'
- 文档-主题分布保存在 'results/文档主题分布.csv'

## 6. 类别分析
- 按类别统计和关键词分析
- 结果保存在 'results/类别关键词分析.txt'

## 7. 结论
- 通过文本分析，我们发现了文本数据中的主要主题和关键概念
- 不同类别的文本具有不同的关键词特征
- 主题模型揭示了文本集合中的潜在语义结构

## 8. 使用说明
### 如何修改参数:
1. 数据加载: 修改load_sample_data函数中的示例数据或添加文件读取逻辑
2. 文本预处理: 在preprocess_texts函数中添加或修改预处理步骤
3. 关键词提取: 在extract_keywords函数中修改topK参数调整关键词数量
4. 主题建模: 在vectorize_and_topic_modeling函数中修改n_topics参数调整主题数量

### 数据结构说明:
1. texts: 列表，每个元素是一个文本字符串
2. df: DataFrame，包含'文本ID', '文本内容', '类别', '发布日期'等字段
3. segmented_texts: 列表，每个元素是分词后的词语列表
4. word_freq: Counter对象，包含词语及其频次
5. tfidf_matrix: 稀疏矩阵，表示文本的TF-IDF向量
6. topics: 列表，每个元素是一个主题的关键词列表
""".format(
        total_texts=len(df),
        total_categories=len(df['类别'].unique()),
        category_distribution=', '.join([f"{c}: {n}" for c, n in category_counts.items()]),
        n_topics=len(topics)
    )
    
    # 保存报告
    with open('results/文本分析报告.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\n分析报告已生成")
    print("报告保存在 'results/文本分析报告.md'")

# 生成报告
generate_report()

print("\n文本分析完成！所有结果已保存在results文件夹中。")
# %%
