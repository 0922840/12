import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import io

# 设置页面配置
st.set_page_config(page_title="📦 智能订单错峰仿真器", layout="wide")
st.image("https://img.icons8.com/color/96/000000/warehouse.png", width=60)
st.title("📦 智能订单错峰仿真模拟器")
st.markdown("""
<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;'>
    本工具可模拟多策略对订单高峰影响的缓解效果，支持策略优化组合、边际效应分析与仿真可视化。
</div>
""", unsafe_allow_html=True)

# ========================== Sidebar 参数区域 ============================
st.sidebar.header("🔧 模型参数设置")
use_optimized_version = st.sidebar.checkbox("使用优化逻辑", value=True)

# 上传真实订单分布（Excel版）
st.sidebar.markdown("---")
st.sidebar.markdown("📤 上传订单 Excel 文件 (.xlsx)")
uploaded_file = st.sidebar.file_uploader("", type=["xlsx"])

# 默认分布
original_distribution = {
    "09:00–12:00": 0.2,
    "12:00–15:00": 0.2,
    "15:00–18:00": 0.4,
    "18:00–24:00": 0.2
}

total_orders = 10000  # 初始默认值

if uploaded_file:
    try:
        df_raw = pd.read_excel(uploaded_file)
        st.write("✅ 文件成功上传并读取如下数据：")
        st.dataframe(df_raw.head())

        required_cols = {'下单时间', '订单总出库量'}
        if required_cols.issubset(df_raw.columns):
            df_raw['下单时间'] = pd.to_datetime(df_raw['下单时间'], format='%H:%M', errors='coerce')
            df_raw = df_raw.dropna(subset=['下单时间', '订单总出库量'])
            df_raw['时段'] = df_raw['下单时间'].dt.hour

            # 归类时段
            bins = [0, 12, 15, 18, 24]
            labels = ['09:00–12:00', '12:00–15:00', '15:00–18:00', '18:00–24:00']
            df_raw['时段分类'] = pd.cut(df_raw['时段'], bins=bins, labels=labels, right=False)
            # ✅ 修复订单总量偏差问题
            total_orders = int(df_raw['订单总出库量'].sum())  # 使用整个文件的订单总出库量
            df_valid = df_raw[df_raw['时段分类'].notna()]  # 仅用于分布结构
            distribution_series = df_valid.groupby('时段分类')['订单总出库量'].sum()
            distribution_series = distribution_series / distribution_series.sum()  # 归一化分布结构
            original_distribution = distribution_series.to_dict()
        else:
            st.sidebar.warning(f"⚠️ 缺少必要字段: {required_cols - set(df_raw.columns)}")
    except Exception as e:
        st.sidebar.warning(f"📛 文件处理失败: {e}")
else:
    total_orders = st.sidebar.number_input("请输入总订单量", min_value=1000, max_value=100000, value=10000, step=1000)

# 策略设置
st.sidebar.markdown("---")
discount_shift_rate = st.sidebar.slider("错峰折扣转移比例（%）", 0.0, 50.0, 15.0, step=0.5) / 100
vip_shift_rate = st.sidebar.slider("会员预约转移比例（%）", 0.0, 50.0, 10.5, step=0.5) / 100
reminder_shift_rate = st.sidebar.slider("智能提醒转移比例（%）", 0.0, 50.0, 6.0, step=0.5) / 100

# ========================== 模拟函数区 ============================
def initialize_orders(total_orders):
    return {t: int(total_orders * p) for t, p in original_distribution.items()}

def apply_strategies_with_diminishing_effect(total_orders, original_distribution, adjusted_orders, *rates):
    strategies = [
        {"name": "错峰折扣", "from": "15:00–18:00", "to": "09:00–12:00", "rate": rates[0]},
        {"name": "会员预约", "from": "15:00–18:00", "to": "12:00–15:00", "rate": rates[1]},
        {"name": "智能提醒", "from": "15:00–18:00", "to": "18:00–24:00", "rate": rates[2]},
    ]
    current_from = adjusted_orders.get("15:00–18:00", 0)
    for s in strategies:
        shift_base = current_from
        shift_amount = int(shift_base * s['rate'])
        shift_amount = min(shift_amount, adjusted_orders[s['from']])
        adjusted_orders[s['to']] += shift_amount
        adjusted_orders[s['from']] -= shift_amount
        current_from -= shift_amount
    return adjusted_orders

# ========================== 订单处理 ============================
if use_optimized_version:
    st.markdown("### 🚀 使用优化逻辑")
    adjusted_orders = initialize_orders(total_orders)
    adjusted_orders = apply_strategies_with_diminishing_effect(total_orders, original_distribution, adjusted_orders,
                                                               discount_shift_rate, vip_shift_rate, reminder_shift_rate)
else:
    st.markdown("### 🛠️ 使用原始逻辑")
    adjusted_orders = initialize_orders(total_orders)
    for rate in [discount_shift_rate, vip_shift_rate, reminder_shift_rate]:
        shift_amount = int(total_orders * original_distribution.get("15:00–18:00", 0) * rate)
        shift_amount = min(shift_amount, adjusted_orders.get("15:00–18:00", 0))
        adjusted_orders["09:00–12:00"] += shift_amount
        adjusted_orders["15:00–18:00"] -= shift_amount

# ========================== 展示表格与指标 ============================
df = pd.DataFrame.from_dict(adjusted_orders, orient='index', columns=['订单量'])
df['占比'] = (df['订单量'] / total_orders).round(4)
st.subheader("📊 调整后订单分布")
st.dataframe(df)

# ========================== 可视化图表 ============================
st.subheader("📈 订单分布对比图")
labels = list(original_distribution.keys())
original_values = [int(v * total_orders) for v in original_distribution.values()]
adjusted_values = [adjusted_orders.get(t, 0) for t in labels]

fig = go.Figure()
fig.add_trace(go.Bar(x=labels, y=original_values, name='原始', marker_color='indianred'))
fig.add_trace(go.Bar(x=labels, y=adjusted_values, name='调整后', marker_color='seagreen'))

fig.update_layout(
    barmode='group',
    xaxis_title='时段',
    yaxis_title='订单量',
    title='订单高峰调整对比',
    plot_bgcolor='rgba(245, 245, 245, 1)',
    paper_bgcolor='white',
    font=dict(size=14),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

st.plotly_chart(fig)

# ========================== 错峰效果指标 ============================
st.markdown("### 📉 错峰效果分析")
original_peak = original_values[2] if len(original_values) > 2 else 0
adjusted_peak = adjusted_values[2] if len(adjusted_values) > 2 else 0
reduction_ratio = 1 - adjusted_peak / original_peak if original_peak else 0

st.metric("高峰期订单减少率", f"{reduction_ratio * 100:.2f}%")
st.markdown("> 高峰期错峰效果越高，说明系统越能减轻爆单压力。")

# ========================== 图表扩展部分 ============================
st.subheader("📉 高峰订单压力缓解分析")
st.markdown("本图展示高峰时段订单在错峰策略作用下的缓解幅度。")

# 高峰前后订单对比
reduction_value = original_values[2] - adjusted_values[2]
reduction_percent = (reduction_value / original_values[2] * 100) if original_values[2] else 0

fig_reduction = go.Figure()
fig_reduction.add_trace(go.Indicator(
    mode="number+delta",
    value=adjusted_values[2],
    delta={"reference": original_values[2], "valueformat": ".0f", "relative": True},
    title={"text": "高峰订单削减后数量"},
    domain={"x": [0, 1], "y": [0, 1]}
))
st.plotly_chart(fig_reduction)

st.subheader("📌 各策略转移效果表")
st.markdown("表格展示各策略在模拟中的订单转移量和占比情况。")

# 分别计算三项策略的转移量
peak_original = original_values[2]
s1 = int(peak_original * discount_shift_rate)
s2 = int((peak_original - s1) * vip_shift_rate)
s3 = int((peak_original - s1 - s2) * reminder_shift_rate)
total_shift = s1 + s2 + s3

df_strategy = pd.DataFrame({
    "策略": ["错峰折扣", "会员预约", "智能提醒"],
    "转移订单量": [s1, s2, s3],
    "占总转移比例": [s1/total_shift, s2/total_shift, s3/total_shift]
})
df_strategy["占总转移比例"] = (df_strategy["占总转移比例"] * 100).round(2).astype(str) + "%"

st.dataframe(df_strategy)



# ========================== 上传说明 ============================
st.markdown("---")
st.markdown("### 📂 如何上传真实订单数据？")
st.markdown("""
- 请上传Excel（.xlsx）文件，文件需包含如下字段：
    - `下单日期`、`下单时间`、`订单编号`、`订单总出库量`、SKU列、订单类别等
- 工具将根据 `下单时间` 字段自动归类订单时段，并以 `订单总出库量` 作为计算依据。
- 时间格式推荐为 `HH:MM`（例如 `14:30`）。
- 若文件读取成功，你将看到前几行订单样例展示。
""")

st.success("✨ 以上内容已全部支持，可立即运行！")




# ========================== 导出功能 ============================
st.subheader("📤 数据导出")
st.markdown("用户可将模拟结果导出为 Excel 表格或图表截图，用于汇报与归档。")

# Excel 导出
df_export = df.copy()
df_export.reset_index(inplace=True)
df_export.rename(columns={"index": "时段"}, inplace=True)

strategy_export = df_strategy.copy()

export_excel = io.BytesIO()
with pd.ExcelWriter(export_excel, engine="xlsxwriter") as writer:
    df_export.to_excel(writer, sheet_name="订单分布", index=False)
    strategy_export.to_excel(writer, sheet_name="策略转移分析", index=False)
    writer.save()
st.download_button(
    label="📥 下载模拟结果（Excel）",
    data=export_excel.getvalue(),
    file_name="错峰模拟结果.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# 图表导出（Plotly图转图像）
st.markdown("可右键图表保存为PNG图片，或使用浏览器自带截图工具截取图形。")


# ========================== 图3-1：原始 vs 调整订单曲线图 ============================
st.subheader("📈 图3-1 原始与调整后订单分布曲线图")
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=labels, y=original_values, mode='lines+markers', name='原始订单', line=dict(color='indianred')))
fig_line.add_trace(go.Scatter(x=labels, y=adjusted_values, mode='lines+markers', name='调整后订单', line=dict(color='seagreen')))
fig_line.update_layout(title="图3-1 原始与调整后订单趋势对比", xaxis_title="时段", yaxis_title="订单量")
st.plotly_chart(fig_line)

# ========================== 图3-2：策略强度 vs 转移比例（模拟曲线） ============================
st.subheader("📈 图3-2 错峰折扣强度与分流比例关系")
import numpy as np
r_values = np.linspace(0, 0.5, 50)
rho_values = 1 - np.exp(-6 * r_values)  # 模拟用户对折扣的响应函数
fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=r_values*100, y=rho_values*100, mode='lines', name='折扣转移函数', line=dict(color='royalblue')))
fig_curve.update_layout(title="图3-2 折扣率与分流比例关系", xaxis_title="折扣比例（%）", yaxis_title="预计转移比例（%）")
st.plotly_chart(fig_curve)

# ========================== 图3-3：高峰订单缓解柱状图 ============================
st.subheader("📊 图3-3 高峰订单缓解幅度")
reduction = original_values[2] - adjusted_values[2]
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=["原始高峰订单", "调整后高峰订单"], y=[original_values[2], adjusted_values[2]], marker_color=["orangered", "mediumseagreen"]))
fig_bar.update_layout(title="图3-3 高峰订单量缓解效果", yaxis_title="订单量")
st.plotly_chart(fig_bar)


# ========================== 导出功能 ============================
st.subheader("📤 数据导出")
st.markdown("用户可将模拟结果导出为 Excel 表格，用于汇报与归档。")

df_export = df.copy()
df_export.reset_index(inplace=True)
df_export.rename(columns={"index": "时段"}, inplace=True)

strategy_export = df_strategy.copy()

export_excel = io.BytesIO()
with pd.ExcelWriter(export_excel, engine="xlsxwriter") as writer:
    df_export.to_excel(writer, sheet_name="订单分布", index=False)
    strategy_export.to_excel(writer, sheet_name="策略转移分析", index=False)
export_excel.seek(0)

st.download_button(
    label="📥 下载模拟结果（Excel）",
    data=export_excel,
    file_name="错峰模拟结果.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
