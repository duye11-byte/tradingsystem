#!/usr/bin/env python3
"""
BTC量化交易策略可视化报告生成器
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import json
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置高分辨率
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def load_data():
    """加载数据"""
    # 加载回测报告
    with open('btc_quant_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # 加载权益曲线
    equity_df = pd.read_csv('equity_curves.csv', index_col=0, parse_dates=True)
    
    return report, equity_df


def plot_equity_curves(equity_df, save_path):
    """绘制权益曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 选择主要策略
    main_strategies = ['组合策略', 'MVRV策略', '超级趋势策略', 'MACD策略']
    
    # 1. 主要策略权益曲线
    ax1 = axes[0, 0]
    for strategy in main_strategies:
        if strategy in equity_df.columns:
            ax1.plot(equity_df.index, equity_df[strategy], label=strategy, linewidth=2)
    ax1.set_title('主要策略权益曲线对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('账户权益 (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 所有策略收益对比
    ax2 = axes[0, 1]
    if '组合策略' in equity_df.columns:
        # 计算收益率
        returns = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
        returns = returns.sort_values(ascending=True)
        
        colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
        bars = ax2.barh(range(len(returns)), returns.values, color=colors)
        ax2.set_yticks(range(len(returns)))
        ax2.set_yticklabels(returns.index, fontsize=9)
        ax2.set_xlabel('总收益率 (%)')
        ax2.set_title('各策略总收益率对比', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, returns.values)):
            ax2.text(val + (2 if val >= 0 else -2), i, f'{val:.1f}%', 
                    va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    
    # 3. 回撤分析
    ax3 = axes[1, 0]
    if '组合策略' in equity_df.columns:
        equity = equity_df['组合策略']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        ax3.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='#e74c3c')
        ax3.plot(equity_df.index, drawdown, color='#e74c3c', linewidth=1.5)
        ax3.set_title('组合策略回撤分析', fontsize=14, fontweight='bold')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('回撤 (%)')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 标注最大回撤
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown.min()
        ax3.annotate(f'最大回撤: {max_dd:.2f}%', 
                    xy=(max_dd_idx, max_dd),
                    xytext=(max_dd_idx, max_dd + 5),
                    fontsize=10, color='#e74c3c',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    # 4. 风险收益散点图
    ax4 = axes[1, 1]
    with open('btc_quant_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    results = report.get('all_results', {})
    if results:
        names = list(results.keys())
        returns = [results[n]['total_return'] * 100 for n in names]
        drawdowns = [results[n]['max_drawdown'] * 100 for n in names]
        sharpes = [results[n]['sharpe_ratio'] for n in names]
        
        scatter = ax4.scatter(drawdowns, returns, c=sharpes, cmap='RdYlGn', 
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('夏普比率')
        
        # 标注重要策略
        for i, name in enumerate(names):
            if returns[i] > 100 or drawdowns[i] < 15:
                ax4.annotate(name, (drawdowns[i], returns[i]), 
                           fontsize=8, ha='center', va='bottom')
        
        ax4.set_xlabel('最大回撤 (%)')
        ax4.set_ylabel('总收益率 (%)')
        ax4.set_title('风险-收益分布图', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加参考线
        ax4.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='目标收益200%')
        ax4.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='风险限制30%')
        ax4.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"权益曲线图已保存: {save_path}")


def plot_strategy_comparison(report, save_path):
    """绘制策略对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    results = report.get('all_results', {})
    
    if not results:
        print("没有回测结果数据")
        return
    
    names = list(results.keys())
    
    # 1. 夏普比率对比
    ax1 = axes[0, 0]
    sharpes = [results[n]['sharpe_ratio'] for n in names]
    colors = ['#2ecc71' if s > 0.5 else '#f39c12' if s > 0 else '#e74c3c' for s in sharpes]
    bars = ax1.bar(range(len(names)), sharpes, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('夏普比率')
    ax1.set_title('各策略夏普比率对比', fontsize=14, fontweight='bold')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='优秀线')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='及格线')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 胜率对比
    ax2 = axes[0, 1]
    win_rates = [results[n]['win_rate'] * 100 for n in names]
    colors = ['#2ecc71' if w > 50 else '#e74c3c' for w in win_rates]
    bars = ax2.bar(range(len(names)), win_rates, color=colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('胜率 (%)')
    ax2.set_title('各策略胜率对比', fontsize=14, fontweight='bold')
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 盈亏比对比
    ax3 = axes[1, 0]
    profit_factors = [results[n]['profit_factor'] for n in names]
    # 限制最大值
    profit_factors = [min(p, 10) for p in profit_factors]
    colors = ['#2ecc71' if p > 1.5 else '#f39c12' if p > 1 else '#e74c3c' for p in profit_factors]
    bars = ax3.bar(range(len(names)), profit_factors, color=colors)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('盈亏比')
    ax3.set_title('各策略盈亏比对比', fontsize=14, fontweight='bold')
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 索提诺比率对比
    ax4 = axes[1, 1]
    sortinos = [results[n]['sortino_ratio'] for n in names]
    colors = ['#2ecc71' if s > 1 else '#f39c12' if s > 0 else '#e74c3c' for s in sortinos]
    bars = ax4.bar(range(len(names)), sortinos, color=colors)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('索提诺比率')
    ax4.set_title('各策略索提诺比率对比', fontsize=14, fontweight='bold')
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='优秀线')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"策略对比图已保存: {save_path}")
    
    # 单独生成雷达图
    plot_radar_chart(report, save_path.replace('.png', '_radar.png'))


def plot_radar_chart(report, save_path):
    """绘制雷达图"""
    qualified = report.get('qualified_strategies', [])
    
    if not qualified:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    top_strategies = qualified[:4]
    
    # 雷达图维度
    categories = ['收益率', '夏普比率', '胜率', '盈亏比', '风险控制']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    colors_radar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, strategy in enumerate(top_strategies):
        values = [
            min(strategy['total_return'] / 5, 1),  # 归一化到0-1
            min(strategy['sharpe_ratio'] / 2, 1),
            strategy['win_rate'],
            min(strategy['profit_factor'] / 3, 1) if strategy['profit_factor'] < 100 else 1,
            1 - strategy['max_drawdown']  # 风险控制得分
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy['name'], color=colors_radar[i])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[i])
    
    ax.set_title('优秀策略综合评分雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"雷达图已保存: {save_path}")


def plot_market_analysis(save_path):
    """绘制市场分析图"""
    # 加载原始数据
    import requests
    
    print("获取BTC价格数据用于市场分析...")
    
    # 获取数据
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": 1520
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    dates = [pd.to_datetime(d[0], unit='ms') for d in data]
    closes = [float(d[4]) for d in data]
    volumes = [float(d[5]) for d in data]
    
    df = pd.DataFrame({
        'date': dates,
        'close': closes,
        'volume': volumes
    })
    df.set_index('date', inplace=True)
    
    # 计算技术指标
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    
    # 计算波动率
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(30).std() * np.sqrt(365) * 100
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # 1. 价格走势与均线
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='BTC价格', linewidth=1.5, color='#2c3e50')
    ax1.plot(df.index, df['sma20'], label='SMA20', linewidth=1, alpha=0.7, color='#3498db')
    ax1.plot(df.index, df['sma50'], label='SMA50', linewidth=1, alpha=0.7, color='#e74c3c')
    ax1.plot(df.index, df['sma200'], label='SMA200', linewidth=1, alpha=0.7, color='#2ecc71')
    
    # 标注重要价格点
    max_price = df['close'].max()
    max_date = df['close'].idxmax()
    min_price = df['close'].min()
    min_date = df['close'].idxmin()
    
    ax1.annotate(f'最高: ${max_price:,.0f}', xy=(max_date, max_price),
                xytext=(max_date, max_price * 1.1),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ecc71'))
    
    ax1.annotate(f'最低: ${min_price:,.0f}', xy=(min_date, min_price),
                xytext=(min_date, min_price * 0.85),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    ax1.set_title('BTC/USDT 价格走势 (2022-2026)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格 (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. 成交量
    ax2 = axes[1]
    colors = ['#2ecc71' if df['close'].iloc[i] >= df['close'].iloc[i-1] else '#e74c3c' 
              for i in range(1, len(df))]
    colors.insert(0, '#2ecc71')
    
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.7, width=1)
    ax2.set_title('成交量分析', fontsize=14, fontweight='bold')
    ax2.set_ylabel('成交量 (BTC)')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 3. 波动率
    ax3 = axes[2]
    ax3.fill_between(df.index, df['volatility'], alpha=0.3, color='#9b59b6')
    ax3.plot(df.index, df['volatility'], color='#9b59b6', linewidth=1.5)
    
    # 标注高波动期
    high_vol_threshold = df['volatility'].quantile(0.9)
    
    ax3.axhline(y=high_vol_threshold, color='#e74c3c', linestyle='--', 
               alpha=0.7, label=f'高波动阈值 ({high_vol_threshold:.1f}%)')
    
    ax3.set_title('年化波动率 (30日滚动)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('波动率 (%)')
    ax3.set_xlabel('日期')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"市场分析图已保存: {save_path}")


def plot_trade_analysis(report, save_path):
    """绘制交易分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    qualified = report.get('qualified_strategies', [])
    
    if not qualified:
        print("没有符合条件的策略")
        return
    
    # 1. 交易次数分布
    ax1 = axes[0, 0]
    results = report.get('all_results', {})
    trade_counts = [results[n]['total_trades'] for n in results.keys()]
    names = list(results.keys())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    ax1.bar(range(len(names)), trade_counts, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('交易次数')
    ax1.set_title('各策略交易次数', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 平均持仓时间
    ax2 = axes[0, 1]
    holding_times = [results[n]['avg_holding_time'] for n in results.keys()]
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(names)))
    ax2.bar(range(len(names)), holding_times, color=colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('平均持仓时间 (小时)')
    ax2.set_title('各策略平均持仓时间', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 年化收益对比
    ax3 = axes[1, 0]
    annual_returns = [results[n]['annual_return'] * 100 for n in results.keys()]
    
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in annual_returns]
    ax3.bar(range(len(names)), annual_returns, color=colors)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('年化收益率 (%)')
    ax3.set_title('各策略年化收益率', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 索提诺比率
    ax4 = axes[1, 1]
    sortinos = [results[n]['sortino_ratio'] for n in results.keys()]
    
    colors = ['#2ecc71' if s > 1 else '#f39c12' if s > 0 else '#e74c3c' for s in sortinos]
    ax4.bar(range(len(names)), sortinos, color=colors)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('索提诺比率')
    ax4.set_title('各策略索提诺比率', fontsize=14, fontweight='bold')
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"交易分析图已保存: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("BTC量化交易策略可视化报告生成")
    print("=" * 60)
    
    # 加载数据
    report, equity_df = load_data()
    
    # 生成图表
    plot_equity_curves(equity_df, 'equity_curves.png')
    plot_strategy_comparison(report, 'strategy_comparison.png')
    plot_market_analysis('market_analysis.png')
    plot_trade_analysis(report, 'trade_analysis.png')
    
    print("\n" + "=" * 60)
    print("可视化报告生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
