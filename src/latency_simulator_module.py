"""
================================================================================
latency_simulator_module.py - Módulo de Simulación de Latencia
================================================================================

Función principal:
- simulate_latency_with_losses: Simula ejecución con latencia, calculando profit/loss en T + Latency

ALGORITMO:
1. Para cada oportunidad detectada:
   - Obtener epoch original
   - Calcular target_epoch = original_epoch + latency_us
   - Consultar quote en target_epoch usando get_quote_at_epoch
   - Obtener precios y cantidades de los exchanges específicos en T + latency
   - Verificar que los precios sean válidos
   - Calcular profit/loss = (sell_price - buy_price) * tradeable_qty
   - Acumular profit total (puede ser negativo = pérdida)

IMPORTANTE:
- Incluye tanto profits como losses
- Usa cantidades disponibles en T + latency (pueden ser menores que en T)
- Si los precios ya no son válidos en T + latency, la oportunidad se pierde
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from data_loader_module import is_valid_price
from consolidator_module import get_quote_at_epoch

# Importar get_quote_at_epoch directamente si no está disponible
try:
    from consolidator_module import get_quote_at_epoch
except ImportError:
    # Fallback: definir aquí si no está disponible
    def get_quote_at_epoch(consolidated, target_epoch, method='nearest'):
        from consolidator_module import get_quote_at_epoch as _get_quote
        return _get_quote(consolidated, target_epoch, method)

logger = logging.getLogger(__name__)


def simulate_latency_with_losses(opportunities: pd.DataFrame, consolidated: pd.DataFrame, latency_us: int) -> float:
    """
    Simula ejecución con latencia, calculando profit/loss en T + Latency.
    
    Incluye tanto profits como losses. Si en T + latency los precios ya no son válidos
    o las cantidades son insuficientes, la oportunidad se pierde o genera pérdida.
    
    Args:
        opportunities: DataFrame con oportunidades detectadas (debe tener columnas:
                      epoch, buy_exchange, sell_exchange, buy_price, sell_price, tradeable_qty)
        consolidated: DataFrame consolidado con epoch como índice
        latency_us: Latencia en microsegundos
        
    Returns:
        Profit total (puede ser negativo = pérdida)
    """
    if opportunities.empty or consolidated.empty:
        return 0.0
    
    # Trabajar con copias
    opps = opportunities.copy()
    
    # Asegurar que consolidated está indexado por epoch
    if consolidated.index.name != 'epoch':
        consolidated = consolidated.set_index('epoch')
    
    total_profit = 0.0
    
    for _, opp in opps.iterrows():
        original_epoch = opp['epoch']
        target_epoch = original_epoch + latency_us
        
        # Obtener quote en T + latency
        quote_at_latency = get_quote_at_epoch(consolidated, target_epoch)
        
        if quote_at_latency is None:
            continue
        
        # Obtener precios de los exchanges específicos en T + latency
        buy_exchange = opp['buy_exchange']
        sell_exchange = opp['sell_exchange']
        
        buy_price_col = f'{buy_exchange}_ask'
        sell_price_col = f'{sell_exchange}_bid'
        buy_qty_col = f'{buy_exchange}_askqty'
        sell_qty_col = f'{sell_exchange}_bidqty'
        
        if buy_price_col not in quote_at_latency.index or sell_price_col not in quote_at_latency.index:
            continue
        
        buy_price = quote_at_latency[buy_price_col]
        sell_price = quote_at_latency[sell_price_col]
        
        # Verificar que los precios sean válidos
        if pd.isna(buy_price) or pd.isna(sell_price):
            continue
        
        if not is_valid_price(buy_price) or not is_valid_price(sell_price):
            continue
        
        # Obtener cantidades
        sell_qty = quote_at_latency.get(sell_qty_col, 0) if sell_qty_col in quote_at_latency.index else 0
        buy_qty = quote_at_latency.get(buy_qty_col, 0) if buy_qty_col in quote_at_latency.index else 0
        
        if pd.isna(sell_qty) or sell_qty <= 0:
            sell_qty = 0
        if pd.isna(buy_qty) or buy_qty <= 0:
            buy_qty = 0
        
        if sell_qty <= 0 or buy_qty <= 0:
            continue
        
        # Usar cantidad original o cantidad disponible, la que sea menor
        original_qty = opp.get('tradeable_qty', 0)
        if pd.isna(original_qty) or original_qty <= 0:
            original_qty = min(sell_qty, buy_qty)
        
        tradeable_qty = min(original_qty, sell_qty, buy_qty)
        
        if tradeable_qty <= 0:
            continue
        
        # Calcular profit/loss (puede ser negativo)
        profit_per_share = sell_price - buy_price
        realized_profit = profit_per_share * tradeable_qty
        
        total_profit += realized_profit
    
    return total_profit
