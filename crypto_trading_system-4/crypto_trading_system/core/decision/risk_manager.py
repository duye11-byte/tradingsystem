"""
风险管理器
管理系统风险，包括仓位风险、日风险、回撤等
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .decision_types import (
    RiskProfile,
    Position,
    PortfolioState,
    DecisionConfig,
    TradingDecision
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    风险管理器
    
    管理系统风险，提供以下功能：
    1. 风险评估
    2. 风险限制检查
    3. 回撤监控
    4. 风险调整建议
    """
    
    def __init__(self, config: DecisionConfig):
        """
        初始化风险管理器
        
        Args:
            config: 决策配置
        """
        self.config = config
        
        # 风险状态
        self.risk_profile = RiskProfile()
        
        # 历史数据
        self.daily_pnl_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []
        
        # 峰值权益
        self.peak_equity: float = 0.0
        self.peak_equity_date: Optional[datetime] = None
        
        # 日风险追踪
        self.current_day: datetime = datetime.now().date()
        self.daily_risk_used: float = 0.0
        
    def assess_risk(
        self,
        portfolio_state: PortfolioState,
        positions: Dict[str, Position]
    ) -> RiskProfile:
        """
        评估当前风险状况
        
        Args:
            portfolio_state: 组合状态
            positions: 持仓字典
            
        Returns:
            RiskProfile: 风险画像
        """
        profile = RiskProfile()
        
        # 计算总敞口
        profile.total_exposure = sum(
            pos.get_notional_value() for pos in positions.values()
        )
        
        # 计算总风险
        profile.total_risk = self._calculate_total_risk(positions)
        
        # 计算持仓统计
        profile.position_count = len(positions)
        profile.long_exposure = sum(
            pos.get_notional_value() for pos in positions.values()
            if pos.side.value == 'long'
        )
        profile.short_exposure = sum(
            pos.get_notional_value() for pos in positions.values()
            if pos.side.value == 'short'
        )
        
        # 计算回撤
        profile.current_drawdown = self._calculate_drawdown(portfolio_state.total_equity)
        profile.max_drawdown_reached = self._get_max_drawdown()
        
        # 检查风险限制
        profile.risk_limit_reached = self._is_risk_limit_reached(profile)
        profile.daily_limit_reached = self._is_daily_limit_reached()
        profile.drawdown_limit_reached = self._is_drawdown_limit_reached(profile)
        
        # 更新风险画像
        self.risk_profile = profile
        
        return profile
    
    def validate_decision(
        self,
        decision: TradingDecision,
        portfolio_state: PortfolioState,
        positions: Dict[str, Position]
    ) -> tuple:
        """
        验证交易决策是否符合风险规则
        
        Args:
            decision: 交易决策
            portfolio_state: 组合状态
            positions: 持仓字典
            
        Returns:
            tuple: (是否通过, 原因列表)
        """
        reasons = []
        
        # 1. 检查单笔交易风险
        if decision.risk_amount > 0:
            risk_pct = decision.risk_amount / portfolio_state.total_equity
            if risk_pct > self.config.max_risk_per_trade:
                reasons.append(
                    f"Risk per trade too high: {risk_pct:.2%} > "
                    f"{self.config.max_risk_per_trade:.2%}"
                )
        
        # 2. 检查日风险限制
        daily_risk = self._get_daily_risk()
        if daily_risk > self.config.max_daily_risk:
            reasons.append(
                f"Daily risk limit reached: {daily_risk:.2%} > "
                f"{self.config.max_daily_risk:.2%}"
        )
        
        # 3. 检查回撤限制
        current_drawdown = self._calculate_drawdown(portfolio_state.total_equity)
        if current_drawdown > self.config.max_drawdown:
            reasons.append(
                f"Max drawdown reached: {current_drawdown:.2%} > "
                f"{self.config.max_drawdown:.2%}"
            )
        
        # 4. 检查风险收益比
        if decision.risk_reward_ratio > 0 and decision.risk_reward_ratio < 1.0:
            reasons.append(
                f"Risk-reward ratio too low: {decision.risk_reward_ratio:.2f} < 1.0"
            )
        
        # 5. 检查总敞口
        total_exposure = sum(pos.get_notional_value() for pos in positions.values())
        new_exposure = decision.quantity * (decision.entry_price or 0)
        total_exposure_ratio = (total_exposure + new_exposure) / portfolio_state.total_equity
        
        if total_exposure_ratio > 2.0:  # 总敞口超过权益的2倍
            reasons.append(
                f"Total exposure too high: {total_exposure_ratio:.2f}x equity"
            )
        
        passed = len(reasons) == 0
        
        if not passed:
            logger.warning(
                f"Decision rejected by risk manager: {reasons}"
            )
        
        return passed, reasons
    
    def calculate_position_risk(
        self,
        position: Position,
        current_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算单个仓位的风险
        
        Args:
            position: 持仓
            current_price: 当前价格 (可选)
            
        Returns:
            Dict: 风险指标
        """
        price = current_price or position.current_price
        
        # 名义价值
        notional = position.quantity * price
        
        # 止损风险
        stop_risk = 0.0
        if position.stop_loss_price:
            stop_distance = abs(price - position.stop_loss_price)
            stop_risk = stop_distance * position.quantity
        
        # 止损风险百分比
        stop_risk_pct = 0.0
        if notional > 0:
            stop_risk_pct = stop_risk / notional
        
        # 波动率风险 (简化计算)
        volatility_risk = notional * 0.02  # 假设2%日波动
        
        return {
            'notional_value': notional,
            'stop_risk': stop_risk,
            'stop_risk_pct': stop_risk_pct,
            'volatility_risk': volatility_risk,
            'total_risk': stop_risk + volatility_risk * 0.5
        }
    
    def update_daily_pnl(self, pnl: float):
        """
        更新日盈亏
        
        Args:
            pnl: 当日盈亏
        """
        today = datetime.now().date()
        
        # 检查是否是新的一天
        if today != self.current_day:
            self.current_day = today
            self.daily_risk_used = 0.0
        
        # 记录盈亏
        self.daily_pnl_history.append({
            'date': today,
            'pnl': pnl,
            'timestamp': datetime.now()
        })
        
        # 更新日风险使用
        if pnl < 0:
            self.daily_risk_used += abs(pnl)
    
    def record_equity(self, equity: float):
        """
        记录权益
        
        Args:
            equity: 当前权益
        """
        # 更新峰值
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_equity_date = datetime.now()
        
        # 记录历史
        self.equity_history.append({
            'equity': equity,
            'timestamp': datetime.now()
        })
        
        # 限制历史大小
        if len(self.equity_history) > 1000:
            self.equity_history = self.equity_history[-1000:]
    
    def get_risk_adjustment(
        self,
        base_position_size: float
    ) -> float:
        """
        根据当前风险状况调整仓位大小
        
        Args:
            base_position_size: 基础仓位大小
            
        Returns:
            float: 调整后的仓位大小
        """
        adjustment = 1.0
        
        # 根据回撤调整
        if self.risk_profile.current_drawdown > 0.05:  # 5%回撤
            adjustment *= 0.8
        if self.risk_profile.current_drawdown > 0.10:  # 10%回撤
            adjustment *= 0.6
        if self.risk_profile.current_drawdown > 0.15:  # 15%回撤
            adjustment *= 0.4
        
        # 根据日风险使用调整
        daily_risk = self._get_daily_risk()
        if daily_risk > self.config.max_daily_risk * 0.5:
            adjustment *= 0.7
        if daily_risk > self.config.max_daily_risk * 0.8:
            adjustment *= 0.5
        
        # 根据持仓数量调整
        if self.risk_profile.position_count >= self.config.max_concurrent_positions:
            adjustment *= 0.5
        
        return base_position_size * adjustment
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        获取风险报告
        
        Returns:
            Dict: 风险报告
        """
        # 计算日盈亏统计
        daily_pnls = [h['pnl'] for h in self.daily_pnl_history]
        
        if daily_pnls:
            avg_daily_pnl = sum(daily_pnls) / len(daily_pnls)
            max_daily_loss = min(daily_pnls)
            max_daily_gain = max(daily_pnls)
        else:
            avg_daily_pnl = 0.0
            max_daily_loss = 0.0
            max_daily_gain = 0.0
        
        # 计算回撤统计
        drawdowns = self._calculate_drawdowns()
        
        return {
            'current_risk_profile': {
                'total_exposure': self.risk_profile.total_exposure,
                'total_risk': self.risk_profile.total_risk,
                'current_drawdown': self.risk_profile.current_drawdown,
                'position_count': self.risk_profile.position_count,
                'risk_limit_reached': self.risk_profile.risk_limit_reached,
                'daily_limit_reached': self.risk_profile.daily_limit_reached,
                'drawdown_limit_reached': self.risk_profile.drawdown_limit_reached
            },
            'daily_statistics': {
                'avg_daily_pnl': avg_daily_pnl,
                'max_daily_loss': max_daily_loss,
                'max_daily_gain': max_daily_gain,
                'trading_days': len(set(h['date'] for h in self.daily_pnl_history))
            },
            'drawdown_statistics': {
                'max_drawdown': self._get_max_drawdown(),
                'current_drawdown': self.risk_profile.current_drawdown,
                'avg_drawdown': sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
            },
            'risk_limits': {
                'max_risk_per_trade': self.config.max_risk_per_trade,
                'max_daily_risk': self.config.max_daily_risk,
                'max_drawdown': self.config.max_drawdown,
                'daily_risk_used': self.daily_risk_used
            }
        }
    
    def _calculate_total_risk(self, positions: Dict[str, Position]) -> float:
        """计算总风险"""
        total_risk = 0.0
        
        for position in positions.values():
            risk = self.calculate_position_risk(position)
            total_risk += risk['total_risk']
        
        return total_risk
    
    def _calculate_drawdown(self, current_equity: float) -> float:
        """计算当前回撤"""
        if self.peak_equity <= 0:
            return 0.0
        
        if current_equity >= self.peak_equity:
            return 0.0
        
        return (self.peak_equity - current_equity) / self.peak_equity
    
    def _calculate_drawdowns(self) -> List[float]:
        """计算历史回撤列表"""
        if not self.equity_history:
            return []
        
        drawdowns = []
        peak = self.equity_history[0]['equity']
        
        for record in self.equity_history:
            equity = record['equity']
            
            if equity > peak:
                peak = equity
            
            if peak > 0:
                drawdown = (peak - equity) / peak
                drawdowns.append(drawdown)
        
        return drawdowns
    
    def _get_max_drawdown(self) -> float:
        """获取最大回撤"""
        drawdowns = self._calculate_drawdowns()
        return max(drawdowns) if drawdowns else 0.0
    
    def _get_daily_risk(self) -> float:
        """获取日风险使用比例"""
        # 这里简化处理，实际需要根据权益计算
        return self.daily_risk_used / 100000 if self.daily_risk_used > 0 else 0.0
    
    def _is_risk_limit_reached(self, profile: RiskProfile) -> bool:
        """检查是否达到风险限制"""
        # 检查总风险
        # 这里简化处理
        return False
    
    def _is_daily_limit_reached(self) -> bool:
        """检查是否达到日风险限制"""
        daily_risk = self._get_daily_risk()
        return daily_risk >= self.config.max_daily_risk
    
    def _is_drawdown_limit_reached(self, profile: RiskProfile) -> bool:
        """检查是否达到回撤限制"""
        return profile.current_drawdown >= self.config.max_drawdown
