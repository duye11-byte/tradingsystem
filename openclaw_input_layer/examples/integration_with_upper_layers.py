"""
OpenClaw 5层系统集成示例
展示如何将第1层（输入层）与第2-5层连接
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, '/mnt/okcomputer/output/openclaw_input_layer')

# 导入第1层
from core.input import (
    InputEngine, InputEngineConfig, InputMode,
    MarketData, PriceData, FearGreedIndex, FundingRateData
)


# ==================== 模拟第2层：推理层 ====================

class TradingSignal:
    """交易信号"""
    def __init__(
        self,
        symbol: str,
        signal_type: str,  # BUY, SELL, HOLD
        confidence: float,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        reasoning: str,
        metadata: Dict[str, Any] = None
    ):
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reasoning = reasoning
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class ReasoningEngine:
    """
    第2层：推理层（模拟）
    实际实现请参考 crypto_trading_system/core/reasoning/
    """
    
    def __init__(self):
        self.name = "ReasoningEngine"
    
    async def process(self, market_data: MarketData) -> Optional[TradingSignal]:
        """
        处理市场数据，生成交易信号
        
        这是简化的实现，实际应该使用：
        - Chain-of-Thought 推理
        - 多模型集成
        - 一致性验证
        """
        if not market_data or not market_data.price_data:
            return None
        
        symbol = market_data.symbol
        price = market_data.price_data.close_price
        
        # 基于恐惧贪婪指数的信号
        fear_greed_score = 50
        if market_data.fear_greed:
            fear_greed_score = market_data.fear_greed.value
        
        # 基于资金费率的信号
        funding_signal = 0
        if market_data.funding_rate:
            funding_rate = float(market_data.funding_rate.funding_rate)
            funding_signal = -funding_rate * 100  # 负费率 = 看涨
        
        # 综合判断
        composite_score = fear_greed_score + funding_signal * 10
        
        # 生成信号
        if composite_score < 30:  # 极度恐慌
            signal_type = "BUY"
            confidence = min(0.9, 0.5 + (30 - composite_score) / 100)
            reasoning = f"极度恐慌 ({fear_greed_score}) + 负资金费率，逆向买入机会"
            stop_loss = price * Decimal("0.95")
            take_profit = price * Decimal("1.10")
        
        elif composite_score > 80:  # 极度贪婪
            signal_type = "SELL"
            confidence = min(0.9, 0.5 + (composite_score - 80) / 100)
            reasoning = f"极度贪婪 ({fear_greed_score})，考虑获利了结"
            stop_loss = price * Decimal("1.05")
            take_profit = price * Decimal("0.90")
        
        else:
            signal_type = "HOLD"
            confidence = 0.5
            reasoning = "市场情绪中性，观望"
            stop_loss = price * Decimal("0.95")
            take_profit = price * Decimal("1.05")
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            metadata={
                'fear_greed': fear_greed_score,
                'funding_rate': float(market_data.funding_rate.funding_rate) if market_data.funding_rate else 0,
                'composite_score': composite_score
            }
        )


# ==================== 模拟第3层：特征工程层 ====================

class FeatureVector:
    """特征向量"""
    def __init__(self, features: Dict[str, float]):
        self.features = features
        self.timestamp = datetime.now()


class FeatureEngine:
    """
    第3层：特征工程层（模拟）
    实际实现请参考 crypto_trading_system-2/core/features/
    """
    
    def __init__(self):
        self.name = "FeatureEngine"
    
    async def extract_features(self, market_data: MarketData) -> FeatureVector:
        """
        从市场数据中提取特征
        
        实际应该包括：
        - 技术指标（RSI, MACD, 布林带等）
        - 链上指标（交易所流向、TVL等）
        - 情绪指标（恐惧贪婪、资金费率等）
        - 复合特征
        """
        features = {}
        
        # 价格特征
        if market_data.price_data:
            features['price'] = float(market_data.price_data.close_price)
            features['price_change_24h'] = float(market_data.price_data.price_change_pct)
        
        # 订单簿特征
        if market_data.orderbook_data:
            features['spread'] = float(market_data.orderbook_data.spread)
            features['imbalance'] = float(market_data.orderbook_data.imbalance)
        
        # 情绪特征
        if market_data.fear_greed:
            features['fear_greed'] = market_data.fear_greed.value
        
        if market_data.funding_rate:
            features['funding_rate'] = float(market_data.funding_rate.funding_rate)
        
        if market_data.long_short_ratio:
            features['long_short_ratio'] = float(market_data.long_short_ratio.long_short_ratio)
        
        # 综合情绪分数
        features['composite_sentiment'] = market_data.composite_sentiment
        
        return FeatureVector(features)


# ==================== 模拟第4层：决策层 ====================

class Decision:
    """决策"""
    def __init__(
        self,
        symbol: str,
        action: str,  # EXECUTE, HOLD, REJECT
        signal: TradingSignal,
        position_size: Decimal,
        risk_level: str,
        metadata: Dict[str, Any] = None
    ):
        self.symbol = symbol
        self.action = action
        self.signal = signal
        self.position_size = position_size
        self.risk_level = risk_level
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class DecisionEngine:
    """
    第4层：决策层（模拟）
    实际实现请参考 crypto_trading_system-3/core/decision/
    """
    
    def __init__(self):
        self.name = "DecisionEngine"
        self.max_position_size = Decimal("0.1")  # 最大10%仓位
        self.min_confidence = 0.6
    
    async def process_signal(self, signal: TradingSignal) -> Decision:
        """
        处理交易信号，生成决策
        
        实际应该包括：
        - 信号验证
        - 仓位管理
        - 风险评估
        - 执行优化
        """
        # 置信度检查
        if signal.confidence < self.min_confidence:
            return Decision(
                symbol=signal.symbol,
                action="REJECT",
                signal=signal,
                position_size=Decimal("0"),
                risk_level="LOW",
                metadata={'reason': '置信度不足'}
            )
        
        # 计算仓位大小（基于置信度）
        position_size = self.max_position_size * Decimal(str(signal.confidence))
        
        # 风险评估
        risk_reward_ratio = (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)
        
        if risk_reward_ratio < 1.5:
            risk_level = "HIGH"
        elif risk_reward_ratio < 2.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return Decision(
            symbol=signal.symbol,
            action="EXECUTE",
            signal=signal,
            position_size=position_size,
            risk_level=risk_level,
            metadata={
                'risk_reward_ratio': float(risk_reward_ratio),
                'expected_return': float(signal.take_profit - signal.entry_price),
                'max_loss': float(signal.entry_price - signal.stop_loss)
            }
        )
    
    async def create_order(self, decision: Decision) -> Dict[str, Any]:
        """创建订单"""
        return {
            'symbol': decision.symbol,
            'side': decision.signal.signal_type,
            'type': 'LIMIT',
            'price': float(decision.signal.entry_price),
            'quantity': float(decision.position_size),
            'stop_loss': float(decision.signal.stop_loss),
            'take_profit': float(decision.signal.take_profit),
            'timestamp': datetime.now().isoformat()
        }


# ==================== 模拟第5层：反馈层 ====================

class TradeResult:
    """交易结果"""
    def __init__(
        self,
        trade_id: str,
        symbol: str,
        entry_price: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
        pnl_percent: Decimal,
        metadata: Dict[str, Any] = None
    ):
        self.trade_id = trade_id
        self.symbol = symbol
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.pnl = pnl
        self.pnl_percent = pnl_percent
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class FeedbackEngine:
    """
    第5层：反馈层（模拟）
    实际实现请参考 crypto_trading_system-4/core/feedback/
    """
    
    def __init__(self):
        self.name = "FeedbackEngine"
        self.trade_history = []
    
    async def process_trade(self, trade_result: TradeResult):
        """
        处理交易结果
        
        实际应该包括：
        - 性能分析
        - 在线学习
        - RLHF训练
        - 反馈存储
        """
        self.trade_history.append(trade_result)
        
        logger.info(f"交易结果: {trade_result.symbol}")
        logger.info(f"  盈亏: ${trade_result.pnl:,.2f} ({trade_result.pnl_percent:.2%})")
        
        # 计算累计收益
        total_pnl = sum(t.pnl for t in self.trade_history)
        win_rate = sum(1 for t in self.trade_history if t.pnl > 0) / len(self.trade_history)
        
        logger.info(f"累计盈亏: ${total_pnl:,.2f}")
        logger.info(f"胜率: {win_rate:.1%}")
        logger.info(f"总交易数: {len(self.trade_history)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.trade_history:
            return {}
        
        pnls = [t.pnl for t in self.trade_history]
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(pnls),
            'max_profit': max(pnls),
            'max_loss': min(pnls)
        }


# ==================== 5层集成 ====================

class OpenClawTradingSystem:
    """
    OpenClaw 5层交易系统集成
    
    Layer 1: 输入层 (Input Layer) - 多源数据融合
    Layer 2: 推理层 (Reasoning Layer) - AI 推理
    Layer 3: 特征工程层 (Feature Layer) - 特征提取
    Layer 4: 决策层 (Decision Layer) - 交易决策
    Layer 5: 反馈层 (Feedback Layer) - 性能分析
    """
    
    def __init__(self):
        # 第1层：输入层
        self.input_engine = InputEngine()
        
        # 第2层：推理层
        self.reasoning_engine = ReasoningEngine()
        
        # 第3层：特征工程层
        self.feature_engine = FeatureEngine()
        
        # 第4层：决策层
        self.decision_engine = DecisionEngine()
        
        # 第5层：反馈层
        self.feedback_engine = FeedbackEngine()
        
        self.running = False
    
    async def start(self):
        """启动系统"""
        logger.info("=" * 60)
        logger.info("OpenClaw 5层交易系统启动")
        logger.info("=" * 60)
        
        await self.input_engine.start()
        self.running = True
        
        logger.info("系统启动完成")
        logger.info("-" * 60)
    
    async def stop(self):
        """停止系统"""
        self.running = False
        await self.input_engine.stop()
        logger.info("系统已停止")
    
    async def process_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        处理单个交易对
        
        完整的数据流：
        Layer 1 -> Layer 3 -> Layer 2 -> Layer 4
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'layers': {}
        }
        
        # Layer 1: 获取市场数据
        logger.info(f"\n[{symbol}] Layer 1: 获取市场数据...")
        input_result = await self.input_engine.get_market_data(
            symbol,
            data_types=['price', 'sentiment']
        )
        
        if not input_result.success:
            logger.error(f"获取数据失败: {input_result.message}")
            return result
        
        market_data = input_result.market_data
        
        if market_data.price_data:
            logger.info(f"  价格: ${market_data.price_data.close_price:,.2f}")
        
        if market_data.fear_greed:
            logger.info(f"  恐惧贪婪: {market_data.fear_greed.value} ({market_data.fear_greed.classification})")
        
        result['layers']['input'] = {
            'success': True,
            'processing_time_ms': input_result.processing_time_ms
        }
        
        # Layer 3: 特征工程
        logger.info(f"[{symbol}] Layer 3: 特征工程...")
        features = await self.feature_engine.extract_features(market_data)
        logger.info(f"  提取 {len(features.features)} 个特征")
        result['layers']['features'] = {
            'feature_count': len(features.features),
            'features': features.features
        }
        
        # Layer 2: AI 推理
        logger.info(f"[{symbol}] Layer 2: AI 推理...")
        signal = await self.reasoning_engine.process(market_data)
        
        if signal:
            logger.info(f"  信号: {signal.signal_type} (置信度: {signal.confidence:.1%})")
            logger.info(f"  理由: {signal.reasoning}")
            result['layers']['reasoning'] = {
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            }
        
        # Layer 4: 决策
        if signal:
            logger.info(f"[{symbol}] Layer 4: 决策...")
            decision = await self.decision_engine.process_signal(signal)
            logger.info(f"  决策: {decision.action}")
            logger.info(f"  仓位: {decision.position_size:.1%}")
            logger.info(f"  风险: {decision.risk_level}")
            result['layers']['decision'] = {
                'action': decision.action,
                'position_size': float(decision.position_size),
                'risk_level': decision.risk_level
            }
            
            # 模拟执行和反馈
            if decision.action == "EXECUTE":
                # 模拟交易结果
                import random
                pnl = Decimal(str(random.uniform(-500, 1000)))
                
                trade_result = TradeResult(
                    trade_id=f"trade_{datetime.now().timestamp()}",
                    symbol=symbol,
                    entry_price=signal.entry_price,
                    exit_price=signal.entry_price + pnl,
                    pnl=pnl,
                    pnl_percent=pnl / signal.entry_price
                )
                
                logger.info(f"[{symbol}] Layer 5: 反馈...")
                await self.feedback_engine.process_trade(trade_result)
        
        return result
    
    async def run_trading_cycle(self, symbols: list):
        """运行交易周期"""
        for symbol in symbols:
            if not self.running:
                break
            
            try:
                await self.process_symbol(symbol)
            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            'input_engine': self.input_engine.get_stats(),
            'feedback_engine': self.feedback_engine.get_performance_report()
        }


async def main():
    """主函数"""
    # 创建系统
    system = OpenClawTradingSystem()
    
    try:
        # 启动
        await system.start()
        
        # 运行交易周期
        symbols = ['BTCUSDT', 'ETHUSDT']
        await system.run_trading_cycle(symbols)
        
        # 获取统计
        logger.info("\n" + "=" * 60)
        logger.info("系统统计")
        logger.info("=" * 60)
        stats = system.get_system_stats()
        logger.info(f"输入引擎请求数: {stats['input_engine']['total_requests']}")
        logger.info(f"输入引擎成功率: {stats['input_engine']['success_rate']:.1%}")
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
