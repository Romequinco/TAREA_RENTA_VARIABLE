"""
================================================================================
signal_generator.py - Módulo de Detección de Señales de Arbitraje
================================================================================

CONDICIÓN DE ARBITRAJE:
**Global Max Bid > Global Min Ask**

ALGORITMO (basado en analisis_arbitraje_recuperado.ipynb):
1. Calcular Global Max Bid y Global Min Ask
2. Detectar señal de arbitraje (global_max_bid > global_min_ask)
3. Encontrar qué exchanges tienen el max bid y min ask
4. Calcular profit_per_share, tradeable_qty, total_profit
5. Aplicar Rising Edge Detection (rastrea consumo de liquidez independiente)

RISING EDGE DETECTION (v4 - INDEPENDENT SIDE TRACKING):
Rastrea el consumo de liquidez de la Compra (Bid) y la Venta (Ask) por separado.
- Si el precio/exchange del BID cambia -> Reseteamos consumo del BID.
- Si el precio/exchange del ASK cambia -> Reseteamos consumo del ASK.
- Si un lado NO cambia -> Mantenemos su consumo acumulado anterior.

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from config_module import config

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Detecta oportunidades de arbitraje en el consolidated tape.
    
    El sistema busca instantes donde se puede comprar en un venue y
    vender en otro simultáneamente con profit positivo.
    """
    
    def __init__(self):
        """Inicializa el generador de señales."""
        pass
    
    @staticmethod
    def detect_arbitrage_opportunities(consolidated: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta oportunidades de arbitraje donde Global Max Bid > Global Min Ask.
        
        Basado en la función detect_arbitrage_opportunities del código de referencia.
        
        Args:
            consolidated: DataFrame consolidado con columnas de precios por exchange
                         Formato: epoch, {MIC}_bid, {MIC}_ask, {MIC}_bidqty, {MIC}_askqty
            
        Returns:
            DataFrame con oportunidades detectadas (solo filas donde hay arbitraje)
            Columnas: epoch, global_max_bid, global_min_ask, buy_exchange, sell_exchange,
                     buy_price, sell_price, buy_qty, sell_qty, profit_per_share,
                     tradeable_qty, total_profit
        """
        if consolidated.empty:
            return pd.DataFrame()
        
        # Work with a copy to avoid modifying the original
        consolidated_work = consolidated.copy()
        
        # Get all bid and ask columns (excluir _qty columns)
        bid_cols = [col for col in consolidated_work.columns if col.endswith('_bid')]
        ask_cols = [col for col in consolidated_work.columns if col.endswith('_ask')]
        
        if not bid_cols or not ask_cols:
            return pd.DataFrame()
        
        # Calculate global max bid and min ask
        consolidated_work['global_max_bid'] = consolidated_work[bid_cols].max(axis=1, skipna=True)
        consolidated_work['global_min_ask'] = consolidated_work[ask_cols].min(axis=1, skipna=True)
        
        # Find arbitrage opportunities
        arbitrage_mask = consolidated_work['global_max_bid'] > consolidated_work['global_min_ask']
        opportunities = consolidated_work[arbitrage_mask].copy()
        
        if opportunities.empty:
            return pd.DataFrame()
        
        # Save epoch before resetting index (if it exists as index)
        epoch_from_index = None
        if opportunities.index.name == 'epoch' and 'epoch' not in opportunities.columns:
            epoch_from_index = opportunities.index.values
        
        # Reset index BEFORE applying find_exchanges
        # This ensures both DataFrames have clean numeric indices (0, 1, 2...) for pd.concat
        opportunities = opportunities.reset_index(drop=True)
        
        # Restore epoch if it was in the index
        if epoch_from_index is not None and 'epoch' not in opportunities.columns:
            opportunities['epoch'] = epoch_from_index
        
        # Find which exchanges have the max bid and min ask
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
            
            # Manejar ambos formatos de cantidad: _askqty y _ask_qty
            buy_qty = 0
            sell_qty = 0
            
            if min_ask_exchange:
                buy_qty_col1 = f'{min_ask_exchange}_askqty'
                buy_qty_col2 = f'{min_ask_exchange}_ask_qty'
                if buy_qty_col1 in row.index:
                    buy_qty = row.get(buy_qty_col1, 0)
                elif buy_qty_col2 in row.index:
                    buy_qty = row.get(buy_qty_col2, 0)
            
            if max_bid_exchange:
                sell_qty_col1 = f'{max_bid_exchange}_bidqty'
                sell_qty_col2 = f'{max_bid_exchange}_bid_qty'
                if sell_qty_col1 in row.index:
                    sell_qty = row.get(sell_qty_col1, 0)
                elif sell_qty_col2 in row.index:
                    sell_qty = row.get(sell_qty_col2, 0)
            
            return pd.Series({
                'buy_exchange': min_ask_exchange,
                'sell_exchange': max_bid_exchange,
                'buy_price': row[f'{min_ask_exchange}_ask'] if min_ask_exchange else None,
                'sell_price': row[f'{max_bid_exchange}_bid'] if max_bid_exchange else None,
                'buy_qty': buy_qty,
                'sell_qty': sell_qty
            })
        
        # Apply function to find exchanges (now with clean indices)
        exchange_info = opportunities.apply(find_exchanges, axis=1)
        
        # Both DataFrames now have clean numeric indices, so concat will work perfectly
        opportunities = pd.concat([opportunities, exchange_info], axis=1)
        
        # Calculate profit per share
        opportunities['profit_per_share'] = opportunities['sell_price'] - opportunities['buy_price']
        
        # Calculate tradeable quantity
        opportunities['tradeable_qty'] = opportunities[['buy_qty', 'sell_qty']].min(axis=1)
        
        # Calculate total profit
        opportunities['total_profit'] = opportunities['profit_per_share'] * opportunities['tradeable_qty']
        
        return opportunities
    
    @staticmethod
    def apply_rising_edge(signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        FILTRO DE LIQUIDEZ INDEPENDIENTE (Rising Edge v4) - VERSIÓN FINAL
        
        Basado en apply_rising_edge_detection del código de referencia.
        
        Lógica:
        Rastrea el consumo de liquidez de la Compra (Bid) y la Venta (Ask) por separado.
        
        1. Si el precio/exchange del BID cambia -> Reseteamos consumo del BID.
        2. Si el precio/exchange del ASK cambia -> Reseteamos consumo del ASK.
        3. Si un lado NO cambia -> Mantenemos su consumo acumulado anterior.
        
        Esto evita que un movimiento en BME nos haga "olvidar" que ya agotamos la liquidez en AQUIS.
        
        Args:
            signals_df: DataFrame con oportunidades detectadas (debe tener buy_exchange, sell_exchange,
                       buy_price, sell_price, buy_qty, sell_qty, profit_per_share, tradeable_qty)
            
        Returns:
            DataFrame filtrado con solo rising edges válidos y cantidades ajustadas
        """
        if signals_df.empty:
            return signals_df
        
        # Asegurar orden cronológico
        signals_df = signals_df.sort_values('epoch').reset_index(drop=True)
        rising_edge = []
        
        # --- Estado del último trade (Lado COMPRA) ---
        last_buy_price = -1.0
        last_buy_ex = ""
        consumed_buy_qty = 0
        
        # --- Estado del último trade (Lado VENTA) ---
        last_sell_price = -1.0
        last_sell_ex = ""
        consumed_sell_qty = 0
        
        for idx in range(len(signals_df)):
            opp = signals_df.iloc[idx].copy()
            
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
                # Nueva orden en el mercado -> Reseteamos lo consumido en este lado
                consumed_buy_qty = 0
                last_buy_price = curr_buy_price
                last_buy_ex = curr_buy_ex
                
            # --- LÓGICA DE VENTA (ASK) ---
            # ¿Es una orden de venta distinta a la anterior?
            sell_changed = (curr_sell_ex != last_sell_ex) or \
                           (not np.isclose(curr_sell_price, last_sell_price, rtol=1e-9))
            
            if sell_changed:
                # Nueva orden en el mercado -> Reseteamos lo consumido en este lado
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
                opp['is_rising_edge'] = True  # Marcar como rising edge válido
                
                if 'profit_per_share' in opp:
                    opp['total_profit'] = opp['profit_per_share'] * trade_qty
                
                rising_edge.append(opp)
                
                # --- ACUMULAR CONSUMO ---
                # Sumamos lo operado al acumulado de ambos lados (para que se recuerde en la siguiente iteración)
                consumed_buy_qty += trade_qty
                consumed_sell_qty += trade_qty

        result = pd.DataFrame(rising_edge).reset_index(drop=True)
        
        # Asegurar que is_rising_edge existe en todas las filas
        if len(result) > 0 and 'is_rising_edge' not in result.columns:
            result['is_rising_edge'] = True
        
        return result
    
    def generate_signals(self, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Función principal para generar señales de arbitraje.
        
        Basado en el flujo del código de referencia:
        1. Detecta oportunidades usando detect_arbitrage_opportunities
        2. Aplica rising edge detection avanzado
        3. Retorna DataFrame con señales completas
        
        Args:
            consolidated_df: DataFrame consolidado con columnas:
                           [epoch, XMAD_bid, XMAD_ask, XMAD_bid_qty, XMAD_ask_qty, ...]
            
        Returns:
            DataFrame con señales detectadas
        """
        print("\n" + "=" * 80)
        print("DETECCIÓN DE SEÑALES DE ARBITRAJE")
        print("=" * 80)
        
        if consolidated_df.empty:
            logger.warning("DataFrame consolidado vacío")
            return pd.DataFrame()
        
        # Paso 1: Detectar oportunidades básicas (igual que el código de referencia)
        opportunities = self.detect_arbitrage_opportunities(consolidated_df)
        
        if opportunities.empty:
            print("  No se encontraron oportunidades de arbitraje")
            return pd.DataFrame()
        
        print(f"  Oportunidades detectadas (antes de rising edge): {len(opportunities):,}")
        
        # Paso 2: Aplicar rising edge detection avanzado (igual que el código de referencia)
        opportunities = self.apply_rising_edge(opportunities)
        
        print(f"  Oportunidades después de rising edge: {len(opportunities):,}")
        
        if opportunities.empty:
            return pd.DataFrame()
        
        # Crear DataFrame de señales completo para compatibilidad con código existente
        signals_df = consolidated_df.copy()
        signals_df['signal'] = 0
        signals_df['is_opportunity'] = False
        signals_df['is_rising_edge'] = False
        
        # Marcar oportunidades en el DataFrame completo
        if 'epoch' in opportunities.columns:
            # Merge con el DataFrame consolidado para marcar oportunidades
            opportunities_with_epoch = opportunities.set_index('epoch')
            signals_df = signals_df.set_index('epoch')
            
            # Marcar oportunidades
            for epoch in opportunities_with_epoch.index:
                if epoch in signals_df.index:
                    signals_df.loc[epoch, 'signal'] = 1
                    signals_df.loc[epoch, 'is_opportunity'] = True
                    signals_df.loc[epoch, 'is_rising_edge'] = True
            
            # Añadir columnas de oportunidades
            for col in ['global_max_bid', 'global_min_ask', 'buy_exchange', 'sell_exchange',
                        'buy_price', 'sell_price', 'buy_qty', 'sell_qty', 'profit_per_share',
                        'tradeable_qty', 'total_profit']:
                if col in opportunities_with_epoch.columns:
                    signals_df[col] = None
                    for epoch in opportunities_with_epoch.index:
                        if epoch in signals_df.index:
                            signals_df.loc[epoch, col] = opportunities_with_epoch.loc[epoch, col]
            
            signals_df = signals_df.reset_index()
        
        # Añadir columnas adicionales para compatibilidad
        if 'max_bid' not in signals_df.columns and 'global_max_bid' in signals_df.columns:
            signals_df['max_bid'] = signals_df['global_max_bid']
        if 'min_ask' not in signals_df.columns and 'global_min_ask' in signals_df.columns:
            signals_df['min_ask'] = signals_df['global_min_ask']
        if 'venue_max_bid' not in signals_df.columns and 'sell_exchange' in signals_df.columns:
            signals_df['venue_max_bid'] = signals_df['sell_exchange']
        if 'venue_min_ask' not in signals_df.columns and 'buy_exchange' in signals_df.columns:
            signals_df['venue_min_ask'] = signals_df['buy_exchange']
        
        # Calcular theoretical_profit y executable_qty para compatibilidad
        if 'theoretical_profit' not in signals_df.columns:
            if 'profit_per_share' in signals_df.columns:
                signals_df['theoretical_profit'] = signals_df['profit_per_share']
            else:
                signals_df['theoretical_profit'] = 0.0
        
        if 'executable_qty' not in signals_df.columns:
            if 'tradeable_qty' in signals_df.columns:
                signals_df['executable_qty'] = signals_df['tradeable_qty']
            else:
                signals_df['executable_qty'] = 0.0
        
        # RESUMEN
        total_snapshots = len(consolidated_df)
        total_opportunities = len(opportunities)
        total_profit = opportunities['total_profit'].sum() if len(opportunities) > 0 else 0.0
        
        print(f"\n  RESULTADOS:")
        print(f"    - Total snapshots: {total_snapshots:,}")
        print(f"    - Oportunidades detectadas: {total_opportunities:,}")
        print(f"    - Profit teórico total: €{total_profit:.2f}")
        
        if total_opportunities > 0:
            avg_profit = opportunities['total_profit'].mean()
            max_profit = opportunities['total_profit'].max()
            print(f"    - Profit medio: €{avg_profit:.2f}")
            print(f"    - Profit máximo: €{max_profit:.2f}")
        
        return signals_df
    
    def detect_opportunities(self, 
                            consolidated_tape: pd.DataFrame, 
                            executed_trades: list = None,
                            isin: str = None) -> pd.DataFrame:
        """
        Detecta oportunidades de arbitraje (compatibilidad con código existente).
        
        Similar al flujo del código de referencia:
        1. Detecta oportunidades usando detect_arbitrage_opportunities
        2. Aplica rising edge detection avanzado
        3. Retorna DataFrame con oportunidades filtradas
        
        Args:
            consolidated_tape: DataFrame consolidado
            executed_trades: Lista de trades ejecutados (opcional, no usado aún)
            isin: ISIN procesado (opcional)
            
        Returns:
            DataFrame con oportunidades detectadas y filtradas por rising edge
        """
        # Paso 1: Detectar oportunidades básicas (igual que el código de referencia)
        opportunities = self.detect_arbitrage_opportunities(consolidated_tape)
        
        if opportunities.empty:
            logger.info("No se encontraron oportunidades de arbitraje")
            return pd.DataFrame()
        
        # Paso 2: Aplicar rising edge detection avanzado (igual que el código de referencia)
        opportunities = self.apply_rising_edge(opportunities)
        
        if opportunities.empty:
            logger.info("No quedaron oportunidades después del rising edge detection")
            return pd.DataFrame()
        
        # Añadir columnas adicionales para compatibilidad
        opportunities['signal'] = 1
        opportunities['is_opportunity'] = True
        opportunities['is_rising_edge'] = True
        opportunities['theoretical_profit'] = opportunities.get('profit_per_share', 0)
        opportunities['executable_qty'] = opportunities.get('tradeable_qty', 0)
        
        # Mapear nombres para compatibilidad
        if 'buy_exchange' in opportunities.columns:
            opportunities['venue_min_ask'] = opportunities['buy_exchange']
            opportunities['venue_max_bid'] = opportunities['sell_exchange']
            opportunities['min_ask'] = opportunities['buy_price']
            opportunities['max_bid'] = opportunities['sell_price']
            opportunities['min_ask_qty'] = opportunities['buy_qty']
            opportunities['max_bid_qty'] = opportunities['sell_qty']
            opportunities['global_max_bid'] = opportunities['sell_price']
            opportunities['global_min_ask'] = opportunities['buy_price']
        
        logger.info(f"Detectadas {len(opportunities)} oportunidades después del rising edge")
        return opportunities
    
    @staticmethod
    def validate_signals(signals_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Valida señales y calcula estadísticas (no filtra, solo reporta)."""
        rising_edges = signals_df[signals_df.get('is_rising_edge', False)]
        
        stats = {
            'total_signals': len(rising_edges),
            'valid_signals': len(rising_edges),
            'total_profit': rising_edges['total_profit'].sum() if len(rising_edges) > 0 else 0.0,
            'avg_profit': rising_edges['total_profit'].mean() if len(rising_edges) > 0 else 0.0
        }
        
        logger.info(f"Validación: {stats['valid_signals']} señales válidas, profit total: €{stats['total_profit']:.2f}")
        return signals_df, stats
    
    @staticmethod
    def filter_edge_cases(signals_df: pd.DataFrame, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """Filtra edge cases (actualmente deshabilitado - mantiene todas las oportunidades)."""
        return signals_df


# ============================================================================
# FUNCIONES WRAPPER ESTÁTICAS (según especificaciones)
# ============================================================================

def generate_signals(consolidated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Función principal para generar señales de arbitraje.
    
    Wrapper estático según especificaciones.
    
    Args:
        consolidated_df: DataFrame consolidado
        
    Returns:
        DataFrame con señales detectadas
    """
    generator = SignalGenerator()
    return generator.generate_signals(consolidated_df)
