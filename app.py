# 第一行强制页面配置，无前置代码
import streamlit as st
st.set_page_config(page_title="温州市空气质量可视化平台", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# 修复导入错误：正确导入 silhouette_score
from sklearn.metrics import silhouette_score

# 全局绘图字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
numeric_cols = ['臭氧', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2']
full_cols = numeric_cols + ["CO"]

# 缓存数据清洗函数
@st.cache_data
def data_process():
    # 读取原始csv
    try:
        df = pd.read_csv('cata_6716.csv', encoding='gbk')
    except:
        df = pd.read_csv('cata_6716.csv', encoding='gb2312')
    # 去除重复列
    df = df.loc[:, ~df.columns.duplicated()]
    # 字段重命名
    df.rename(columns={
        '细颗粒物': 'PM2.5',
        '可吸入颗粒物': 'PM10',
        '空气质量指数': 'AQI',
        '空气质量水平': 'AQI等级',
        '臭氧1小时平均': '臭氧',
        '二氧化硫': 'SO2',
        '二氧化氮': 'NO2',
        '一氧化碳': 'CO',
        '日期': 'day',
        '月份': 'month'
    }, inplace=True)
    # 拼接月日生成日期，非法日期自动清除
    df['日期拼接'] = df['month'].astype(str).str.cat(df['day'].astype(str), sep='-')
    df['日期'] = pd.to_datetime('2023-' + df['日期拼接'], errors='coerce')
    df.drop(columns=['日期拼接'], inplace=True)
    df = df.dropna(subset=['日期'])
    # 衍生时间字段
    df['星期'] = df['日期'].dt.dayofweek
    def get_season(month):
        if month in [3,4,5]:
            return '春季'
        elif month in [6,7,8]:
            return '夏季'
        elif month in [9,10,11]:
            return '秋季'
        else:
            return '冬季'
    df['季节'] = df['month'].apply(get_season)
    df['是否周末'] = df['星期'].apply(lambda x: '周末' if x >=5 else '工作日')
    # 3σ异常值剔除
    for col in full_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    df.sort_values('日期', inplace=True)
    df.reset_index(drop=True)
    # 保存预处理文件
    df.to_excel("温州空气质量预处理后数据.xlsx", index=False)
    return df

# 离线批量导出函数（仅按钮点击执行）
def export_offline_file(df):
    # 1 描述统计表格
    stats_df = df[full_cols].describe().round(2)
    stats_df.to_excel("指标统计结果.xlsx")
    # 2 AQI等级统计
    aqi_cnt = df['AQI等级'].value_counts()
    aqi_df = pd.DataFrame({"天数": aqi_cnt, "占比%": round(aqi_cnt/len(df)*100, 2)})
    aqi_df.to_excel("AQI等级分布.xlsx")
    # 3 AQI时序静态图
    plt.figure(figsize=(16,6))
    plt.plot(df['日期'], df['AQI'], c="#1f77b4")
    plt.axhline(50, c="g", ls="--", label="优")
    plt.axhline(100, c="orange", ls="--", label="良")
    plt.axhline(150, c="r", ls="--", label="轻度污染")
    plt.title("温州全年AQI时序趋势")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("AQI静态时序图.png", dpi=300, bbox_inches="tight")
    plt.close()
    # 4 相关性热力图
    plt.figure(figsize=(12,8))
    corr = df[full_cols].corr().round(2)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("指标相关性热力图")
    plt.savefig("热力图.png", dpi=300, bbox_inches="tight")
    plt.close()
    # 5 Kmeans聚类（已修复range、导入错误）
    scaler = StandardScaler()
    X = scaler.fit_transform(df[full_cols])
    sil_scores = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        label = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, label))
    best_k = k_range[np.argmax(sil_scores)]
    km_final = KMeans(n_clusters=best_k, random_state=42)
    df['聚类标签'] = km_final.fit_predict(X)
    cluster_res = df.groupby("聚类标签")[full_cols].mean()
    cluster_res.to_excel("聚类分析结果.xlsx")
    # 6 多元线性回归
    X_reg = df[['PM2.5','PM10','臭氧','SO2','NO2','CO']]
    y_reg = df['AQI']
    lr = LinearRegression()
    lr.fit(X_reg, y_reg)
    coef_df = pd.DataFrame({"变量": X_reg.columns, "回归系数": lr.coef_})
    coef_df.to_excel("回归系数表.xlsx")
    return "所有统计表格、图片导出完成，已保存在当前文件夹！"

# 页面主体逻辑（默认直接加载可视化）
def main():
    df = data_process()
    st.title("🌬️ 温州市空气质量时序交互式可视化平台")
    # 侧边栏：筛选 + 离线导出按钮
    with st.sidebar:
        st.header("数据筛选面板")
        month_sel = st.multiselect("选择月份", sorted(df["month"].unique()), default=sorted(df["month"].unique()))
        level_options = sorted(df["AQI等级"].unique())
        # 修复：默认只选中「良」，不再全选所有等级，消除大量重复红色标签
        level_sel = st.multiselect("空气质量等级", level_options, default=["良"])
        filter_df = df[(df["month"].isin(month_sel)) & (df["AQI等级"].isin(level_sel))]
        st.divider()
        st.header("论文文件导出工具")
        if st.button("一键导出统计表格+静态图表+建模结果"):
            msg = export_offline_file(df)
            st.success(msg)
    # 数据预览表格
    st.subheader("筛选后原始数据")
    st.dataframe(filter_df, use_container_width=True)
    # 交互式可视化标签页
    tab1, tab2, tab3 = st.tabs(["AQI时序趋势", "污染物箱线分布", "相关性热力图"])
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(filter_df["日期"], filter_df["AQI"])
        ax1.set_title("每日AQI变化曲线")
        st.pyplot(fig1)
    with tab2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=filter_df[full_cols], ax=ax2)
        st.pyplot(fig2)
    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10,6))
        corr_data = filter_df[full_cols].corr()
        sns.heatmap(corr_data, annot=True, ax=ax3)
        st.pyplot(fig3)

if __name__ == "__main__":
    main()