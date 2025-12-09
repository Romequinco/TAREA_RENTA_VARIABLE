"""
================================================================================
signal_generator_module.py - Módulo de Detección de Oportunidades de Arbitraje
================================================================================

Funciones principales:
- detect_arbitrage_opportunities: Detecta oportunidades donde Global Max Bid > Global Min Ask
- apply_rising_edge_detection: Aplica filtro de liquidez independiente (Rising Edge v4)

ALGORITMO DE DETECCIÓN:
1. Calcular global_max_bid = max(bid de todos los exchanges)
2. Calcular global_min_ask = min(ask de todos los exchanges)
3. Oportunidad existe si: global_max_bid > global_min_ask
4. Identificar exchanges con max bid y min ask
5. Calcular profit_per_share = sell_price - buy_price
6. Calcular tradeable_qty = min(buy_qty, sell_qty)
7. Calcular total_profit = profit_per_share * tradeable_qty

RISING EDGE DETECTION (v4 - Independent Side Tracking):
Rastrea el consumo de liquidez de Compra (Bid) y Venta (Ask) por separado.
- Si precio/exchange del BID cambia → Resetea consumo del BID
- Si precio/exchange del ASK cambia → Resetea consumo del ASK
- Si un lado NO cambia → Mantiene su consumo acumulado anterior
Esto evita que un movimiento en un exchange haga "olvidar" que ya se agotó la liquidez en otro.
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def detect_arbitrage_opportunities(consolidated: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta oportunidades de arbitraje donde Global Max Bid > Global Min Ask.
    
    Proceso:
    1. Calcula global_max_bid y global_min_ask para cada timestamp
    2. Filtra filas donde global_max_bid > global_min_ask
    3. Identifica exchanges con max bid y min ask
    4. Calcula profit_per_share, tradeable_qty, total_profit
    
    Args:
        consolidated: DataFrame consolidado con columnas {exchange}_bid, {exchange}_ask, etc.
        
    Returns:
        DataFrame con oportunidades detectadas. Columnas:
        - epoch, global_max_bid, global_min_ask
        - buy_exchange, sell_exchange, buy_price, sell_price
        - buy_qty, sell_qty, profit_per_share, tradeable_qty, total_profit
    """
    if consolidated.empty:
        return pd.DataFrame()
    
    # Trabajar con copia para no modificar el original
    consolidated_work = consolidated.copy()
    
    # Obtener todas las columnas bid y ask
    bid_cols = [col for col in consolidated_work.columns if col.endswith('_bid')]
    ask_cols = [col for col in consolidated_work.columns if col.endswith('_ask')]
    
    if not bid_cols or not ask_cols:
        return pd.DataFrame()
    
    # Calcular global max bid y min ask
    consolidated_work['global_max_bid'] = consolidated_work[bid_cols].max(axis=1, skipna=True)
    consolidated_work['global_min_ask'] = consolidated_work[ask_cols].min(axis=1, skipna=True)
    
    # Encontrar oportunidades de arbitraje
    arbitrage_mask = consolidated_work['global_max_bid'] > consolidated_work['global_min_ask']
    opportunities = consolidated_work[arbitrage_mask].copy()
    
    if opportunities.empty:
        return pd.DataFrame()
    
    # Guardar epoch antes de resetear índice (si existe como índice)
    epoch_from_index = None
    if opportunities.index.name == 'epoch' and 'epoch' not in opportunities.columns:
        epoch_from_index = opportunities.index.values
    
    # Resetear índice ANTES de aplicar find_exchanges
    # Esto asegura que ambos DataFrames tengan índices numéricos limpios (0, 1, 2...) para pd.concat
    opportunities = opportunities.reset_index(drop=True)
    
    # Restaurar epoch si estaba en el índice
    if epoch_from_index is not None and 'epoch' not in opportunities.columns:
        opportunities['epoch'] = epoch_from_index
    
    # Encontrar qué exchanges tienen el max bid y min ask
    def find_exchanges(row):
        max_bid_exchange = None
        min_ask_exchange = None
        
        for col in bid_cols:
            if not pd.isna(row[col]) and row[col] == row['global_max_bid']:
                max_bid_exchange = col.replace('_bid', '')
                break
        
        for col in ask_cols:
            if not pd.isna(row[col]) and row[col] == row['global_min_ask']:
                min_ask_exchange = col.replace('_ask', '')
                break
        
        return pd.Series({
            'buy_exchange': min_ask_exchange,
            'sell_exchange': max_bid_exchange,
            'buy_price': row[f'{min_ask_exchange}_ask'] if min_ask_exchange else None,
            'sell_price': row[f'{max_bid_exchange}_bid'] if max_bid_exchange else None,
            'buy_qty': row.get(f'{min_ask_exchange}_askqty', 0) if min_ask_exchange and f'{min_ask_exchange}_askqty' in row.index else 0,
            'sell_qty': row.get(f'{max_bid_exchange}_bidqty', 0) if max_bid_exchange and f'{max_bid_exchange}_bidqty' in row.index else 0
        })
    
    # Aplicar función para encontrar exchanges (ahora con índices limpios)
    exchange_info = opportunities.apply(find_exchanges, axis=1)
    
    # Ambos DataFrames ahora tienen índices numéricos limpios, así que concat funcionará perfectamente
    opportunities = pd.concat([opportunities, exchange_info], axis=1)
    
    # Calcular profit por share
    opportunities['profit_per_share'] = opportunities['sell_price'] - opportunities['buy_price']
    
    # Calcular cantidad tradeable
    opportunities['tradeable_qty'] = opportunities[['buy_qty', 'sell_qty']].min(axis=1)
    
    # Calcular profit total
    opportunities['total_profit'] = opportunities['profit_per_share'] * opportunities['tradeable_qty']
    
    return opportunities


def apply_rising_edge_detection(opportunities: pd.DataFrame) -> pd.DataFrame:
    """
    FILTRO DE LIQUIDEZ INDEPENDIENTE (Rising Edge v4) - VERSIÓN FINAL
    
    Lógica:
    Rastrea el consumo de liquidez de la Compra (Bid) y la Venta (Ask) por separado.
    
    1. Si el precio/exchange del BID cambia → Resetea consumo del BID.
    2. Si el precio/exchange del ASK cambia → Resetea consumo del ASK.
    3. Si un lado NO cambia → Mantiene su consumo acumulado anterior.
    
    Esto evita que un movimiento en BME nos haga "olvidar" que ya agotamos la liquidez en AQUIS.
    
    Args:
        opportunities: DataFrame con oportunidades detectadas
        
    Returns:
        DataFrame con solo las oportunidades válidas (rising edges)
    """
    if opportunities.empty:
        return opportunities
    
    # Asegurar orden cronológico
    opportunities = opportunities.sort_values('epoch').reset_index(drop=True)
    rising_edge = []
    
    # --- Estado del último trade (Lado COMPRA) ---
    last_buy_price = -1.0
    last_buy_ex = ""
    consumed_buy_qty = 0
    
    # --- Estado del último trade (Lado VENTA) ---
    last_sell_price = -1.0
    last_sell_ex = ""
    consumed_sell_qty = 0
    
    for idx in range(len(opportunities)):
        opp = opportunities.iloc[idx].copy()
        
        # Datos actuales
        curr_buy_price = opp['buy_price']
        curr_sell_price = opp['sell_price']
        curr_buy_ex = opp['buy_exchange']
        curr_sell_ex = opp['sell_exchange']
        
        # Cantidades TOTALES del snapshot
        curr_b_qty_total = opp.get('buy_qty', 0)
        curr_s_qty_total = opp.get('sell_qty', 0)
        
        if pd.isna(curr_buy_price) or pd.isna(curr_sell_price):
            continue

        # --- LÓGICA DE COMPRA (BID) ---
        # ¿Es una orden de compra distinta a la anterior?
        buy_changed = (curr_buy_ex != last_buy_ex) or \
                      (not np.isclose(curr_buy_price, last_buy_price, rtol=1e-9))
        
        if buy_changed:
            # Nueva orden en el mercado → Reseteamos lo consumido en este lado
            consumed_buy_qty = 0
            last_buy_price = curr_buy_price
            last_buy_ex = curr_buy_ex
            
        # --- LÓGICA DE VENTA (ASK) ---
        # ¿Es una orden de venta distinta a la anterior?
        sell_changed = (curr_sell_ex != last_sell_ex) or \
                       (not np.isclose(curr_sell_price, last_sell_price, rtol=1e-9))
        
        if sell_changed:
            # Nueva orden en el mercado → Reseteamos lo consumido en este lado
            consumed_sell_qty = 0
            last_sell_price = curr_sell_price
            last_sell_ex = curr_sell_ex
            
        # --- CÁLCULO DE DISPONIBLE NETO ---
        # Si un lado no cambió, 'consumed' se mantiene alto, reduciendo el disponible.
        # Si el volumen del snapshot subió (refill), (total - consumed) dará positivo.
        available_buy = max(0, curr_b_qty_total - consumed_buy_qty)
        available_sell = max(0, curr_s_qty_total - consumed_sell_qty)
        
        # Cruzamos lo que queda libre en ambos lados
        trade_qty = min(available_buy, available_sell)
        
        if trade_qty > 0:
            # ¡Hay match!
            opp['tradeable_qty'] = trade_qty
            opp['buy_qty'] = trade_qty
            opp['sell_qty'] = trade_qty
            
            if 'profit_per_share' in opp:
                opp['total_profit'] = opp['profit_per_share'] * trade_qty
            
            rising_edge.append(opp)
            
            # --- ACUMULAR CONSUMO ---
            # Sumamos lo operado al acumulado de ambos lados (para que se recuerde en la siguiente iteración)
            consumed_buy_qty += trade_qty
            consumed_sell_qty += trade_qty

    return pd.DataFrame(rising_edge).reset_index(drop=True)


# ============================================================================
# Clase SignalGenerator (compatibilidad con código existente)
# ============================================================================

class SignalGenerator:
    """
    Clase wrapper para mantener compatibilidad con código existente.
    
    Internamente usa las funciones del módulo.
    """
    
    def __init__(self):
        """Inicializa el generador de señales."""
        pass
    
    @staticmethod
    def detect_arbitrage_opportunities(consolidated: pd.DataFrame) -> pd.DataFrame:
        """Wrapper para detect_arbitrage_opportunities."""
        return detect_arbitrage_opportunities(consolidated)
    
    @staticmethod
    def apply_rising_edge(opportunities: pd.DataFrame) -> pd.DataFrame:
        """Wrapper para apply_rising_edge_detection."""
        return apply_rising_edge_detection(opportunities)
