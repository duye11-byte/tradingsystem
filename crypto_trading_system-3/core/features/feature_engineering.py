"""
特征工程主入口
整合所有特征提取模块，提供统一的特征工程接口
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import asyncio

from .feature_types import (
    FeatureSet, 
    FeatureConfig, 
    FeatureExtractionResult,
    TechnicalFeatures,
    OnchainFeatures,
    SentimentFeatures,
    CompositeFeatures
)
from .technical.technical_indicators import TechnicalIndicators
from .onchain.onchain_metrics import OnchainMetrics
from .sentiment.sentiment_analyzer import SentimentAnalyzer
from .composite.feature_composer import FeatureComposer

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    特征工程主类
    
    整合技术特征、链上特征、情绪特征和组合特征的提取，
    提供统一的特征工程接口。
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征工程
        
        Args:
            config: 特征配置
        """
        self.config = config or FeatureConfig()
        
        # 初始化各特征模块
        self.technical_indicators = TechnicalIndicators()
        self.onchain_metrics = OnchainMetrics()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_composer = FeatureComposer(n_components=3)
        
        # 特征历史缓存 (用于时间序列分析)
        self.feature_history: Dict[str, List[FeatureSet]] = {}
        self.max_history_size = 100
        
        # 性能统计
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'avg_extraction_time_ms': 0
        }
        
        logger.info("FeatureEngineering initialized")
    
    async def extract_features(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        onchain_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> FeatureExtractionResult:
        """
        提取完整特征集
        
        Args:
            symbol: 交易对符号
            ohlcv_data: OHLCV 数据 DataFrame
            onchain_data: 链上数据 (可选)
            sentiment_data: 情绪数据 (可选)
            
        Returns:
            FeatureExtractionResult: 特征提取结果
        """
        import time
        start_time = time.time()
        
        self.stats['total_extractions'] += 1
        
        try:
            # 创建特征集
            feature_set = FeatureSet(
                symbol=symbol,
                timestamp=datetime.now()
            )
            
            # 提取原始价格数据
            if len(ohlcv_data) > 0:
                latest = ohlcv_data.iloc[-1]
                feature_set.open = float(latest.get('open', 0))
                feature_set.high = float(latest.get('high', 0))
                feature_set.low = float(latest.get('low', 0))
                feature_set.close = float(latest.get('close', 0))
                feature_set.volume = float(latest.get('volume', 0))
            
            features_failed = []
            
            # 1. 提取技术特征
            if self.config.technical_enabled:
                try:
                    feature_set.technical = self.technical_indicators.calculate_all(ohlcv_data)
                except Exception as e:
                    logger.error(f"Technical feature extraction failed: {e}")
                    features_failed.append('technical')
            
            # 2. 提取链上特征
            if self.config.onchain_enabled and onchain_data:
                try:
                    feature_set.onchain = self.onchain_metrics.calculate_all(onchain_data)
                except Exception as e:
                    logger.error(f"Onchain feature extraction failed: {e}")
                    features_failed.append('onchain')
            
            # 3. 提取情绪特征
            if self.config.sentiment_enabled and sentiment_data:
                try:
                    feature_set.sentiment = self.sentiment_analyzer.calculate_all(sentiment_data)
                except Exception as e:
                    logger.error(f"Sentiment feature extraction failed: {e}")
                    features_failed.append('sentiment')
            
            # 4. 提取组合特征
            if self.config.composite_enabled:
                try:
                    # 获取历史特征
                    historical = self.feature_history.get(symbol, [])
                    feature_set.composite = self.feature_composer.calculate_all(
                        feature_set, historical
                    )
                except Exception as e:
                    logger.error(f"Composite feature extraction failed: {e}")
                    features_failed.append('composite')
            
            # 保存到历史缓存
            self._add_to_history(symbol, feature_set)
            
            # 计算特征数量
            features_extracted = (
                len(feature_set.technical.to_dict()) +
                len(feature_set.onchain.to_dict()) +
                len(feature_set.sentiment.to_dict()) +
                len(feature_set.composite.to_dict()) +
                5  # OHLCV
            )
            
            extraction_time = (time.time() - start_time) * 1000
            self._update_avg_time(extraction_time)
            self.stats['successful_extractions'] += 1
            
            logger.info(
                f"Feature extraction completed for {symbol}: "
                f"{features_extracted} features in {extraction_time:.1f}ms"
            )
            
            return FeatureExtractionResult(
                success=True,
                feature_set=feature_set,
                error_message=None,
                extraction_time_ms=extraction_time,
                features_extracted=features_extracted,
                features_failed=features_failed
            )
        
        except Exception as e:
            logger.error(f"Feature extraction failed for {symbol}: {e}")
            self.stats['failed_extractions'] += 1
            
            return FeatureExtractionResult(
                success=False,
                feature_set=None,
                error_message=str(e),
                extraction_time_ms=(time.time() - start_time) * 1000,
                features_extracted=0,
                features_failed=['all']
            )
    
    async def extract_batch(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrency: int = 5
    ) -> List[FeatureExtractionResult]:
        """
        批量特征提取
        
        Args:
            tasks: 任务列表，每个任务包含 symbol, ohlcv_data, onchain_data, sentiment_data
            max_concurrency: 最大并发数
            
        Returns:
            List[FeatureExtractionResult]: 特征提取结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _extract_with_semaphore(task):
            async with semaphore:
                return await self.extract_features(
                    symbol=task['symbol'],
                    ohlcv_data=task['ohlcv_data'],
                    onchain_data=task.get('onchain_data'),
                    sentiment_data=task.get('sentiment_data')
                )
        
        results = await asyncio.gather(
            *[_extract_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(FeatureExtractionResult(
                    success=False,
                    feature_set=None,
                    error_message=str(result),
                    extraction_time_ms=0,
                    features_extracted=0,
                    features_failed=['all']
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def extract_features_sync(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        onchain_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> FeatureExtractionResult:
        """
        同步特征提取 (用于非异步环境)
        
        Args:
            symbol: 交易对符号
            ohlcv_data: OHLCV 数据 DataFrame
            onchain_data: 链上数据 (可选)
            sentiment_data: 情绪数据 (可选)
            
        Returns:
            FeatureExtractionResult: 特征提取结果
        """
        import time
        start_time = time.time()
        
        self.stats['total_extractions'] += 1
        
        try:
            # 创建特征集
            feature_set = FeatureSet(
                symbol=symbol,
                timestamp=datetime.now()
            )
            
            # 提取原始价格数据
            if len(ohlcv_data) > 0:
                latest = ohlcv_data.iloc[-1]
                feature_set.open = float(latest.get('open', 0))
                feature_set.high = float(latest.get('high', 0))
                feature_set.low = float(latest.get('low', 0))
                feature_set.close = float(latest.get('close', 0))
                feature_set.volume = float(latest.get('volume', 0))
            
            features_failed = []
            
            # 1. 提取技术特征
            if self.config.technical_enabled:
                try:
                    feature_set.technical = self.technical_indicators.calculate_all(ohlcv_data)
                except Exception as e:
                    logger.error(f"Technical feature extraction failed: {e}")
                    features_failed.append('technical')
            
            # 2. 提取链上特征
            if self.config.onchain_enabled and onchain_data:
                try:
                    feature_set.onchain = self.onchain_metrics.calculate_all(onchain_data)
                except Exception as e:
                    logger.error(f"Onchain feature extraction failed: {e}")
                    features_failed.append('onchain')
            
            # 3. 提取情绪特征
            if self.config.sentiment_enabled and sentiment_data:
                try:
                    feature_set.sentiment = self.sentiment_analyzer.calculate_all(sentiment_data)
                except Exception as e:
                    logger.error(f"Sentiment feature extraction failed: {e}")
                    features_failed.append('sentiment')
            
            # 4. 提取组合特征
            if self.config.composite_enabled:
                try:
                    historical = self.feature_history.get(symbol, [])
                    feature_set.composite = self.feature_composer.calculate_all(
                        feature_set, historical
                    )
                except Exception as e:
                    logger.error(f"Composite feature extraction failed: {e}")
                    features_failed.append('composite')
            
            # 保存到历史缓存
            self._add_to_history(symbol, feature_set)
            
            features_extracted = (
                len(feature_set.technical.to_dict()) +
                len(feature_set.onchain.to_dict()) +
                len(feature_set.sentiment.to_dict()) +
                len(feature_set.composite.to_dict()) +
                5
            )
            
            extraction_time = (time.time() - start_time) * 1000
            self._update_avg_time(extraction_time)
            self.stats['successful_extractions'] += 1
            
            return FeatureExtractionResult(
                success=True,
                feature_set=feature_set,
                error_message=None,
                extraction_time_ms=extraction_time,
                features_extracted=features_extracted,
                features_failed=features_failed
            )
        
        except Exception as e:
            logger.error(f"Feature extraction failed for {symbol}: {e}")
            self.stats['failed_extractions'] += 1
            
            return FeatureExtractionResult(
                success=False,
                feature_set=None,
                error_message=str(e),
                extraction_time_ms=(time.time() - start_time) * 1000,
                features_extracted=0,
                features_failed=['all']
            )
    
    def _add_to_history(self, symbol: str, feature_set: FeatureSet):
        """添加特征到历史缓存"""
        if symbol not in self.feature_history:
            self.feature_history[symbol] = []
        
        self.feature_history[symbol].append(feature_set)
        
        # 限制缓存大小
        if len(self.feature_history[symbol]) > self.max_history_size:
            self.feature_history[symbol] = self.feature_history[symbol][-self.max_history_size:]
    
    def _update_avg_time(self, new_time: float):
        """更新平均提取时间"""
        n = self.stats['successful_extractions']
        if n <= 0:
            return
        current_avg = self.stats['avg_extraction_time_ms']
        self.stats['avg_extraction_time_ms'] = (current_avg * (n - 1) + new_time) / n
    
    def get_feature_history(self, symbol: str) -> List[FeatureSet]:
        """获取特征历史"""
        return self.feature_history.get(symbol, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats['total_extractions']
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_extractions'] / total 
                if total > 0 else 0
            )
        }
    
    def clear_history(self, symbol: Optional[str] = None):
        """清除历史缓存"""
        if symbol:
            self.feature_history.pop(symbol, None)
        else:
            self.feature_history.clear()
    
    def fit_pca(self, symbol: str):
        """
        为指定交易对拟合 PCA
        
        Args:
            symbol: 交易对符号
        """
        historical = self.feature_history.get(symbol, [])
        if len(historical) >= 30:
            self.feature_composer.fit_pca(historical)
            logger.info(f"PCA fitted for {symbol} with {len(historical)} samples")
        else:
            logger.warning(f"Insufficient data for PCA fitting: {len(historical)} < 30")
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        names = ['open', 'high', 'low', 'close', 'volume']
        
        # 技术特征
        tech = TechnicalFeatures()
        names.extend(tech.to_dict().keys())
        
        # 链上特征
        onchain = OnchainFeatures()
        names.extend(onchain.to_dict().keys())
        
        # 情绪特征
        sentiment = SentimentFeatures()
        names.extend(sentiment.to_dict().keys())
        
        # 组合特征
        composite = CompositeFeatures()
        names.extend(composite.to_dict().keys())
        
        return names
    
    def get_feature_count(self) -> int:
        """获取特征总数"""
        return len(self.get_feature_names())
