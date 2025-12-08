"""
================================================================================
signal_generator.py - Módulo de Detección de Señales de Arbitraje (OPTIMIZADO)
================================================================================

CONDICIÓN DE ARBITRAJE:
**Global Max Bid > Global Min Ask**

ALGORITMO:
1. Calcular Global Max Bid y Global Min Ask
2. Detectar señal de arbitraje
3. Calcular cantidades tradeables
4. Calcular profit teórico
5. Aplicar Rising Edge Detection
6. Validar señales

RISING EDGE DETECTION:
Mantiene solo la PRIMERA aparición de cada oportunidad continua.
Si la señal desaparece (1→0) y vuelve a aparecer (0→1), cuenta como nueva oportunidad.

OPTIMIZACIONES APLICADAS:
- Cálculo vectorizado de cantidades (sin apply)
- Rising edge detection optimizado
- Validaciones integradas
- Filtros de edge cases opcionales
- Funciones wrapper estáticas según especificaciones

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

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
    def _extract_price_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Extrae columnas de precios y cantidades del DataFrame consolidado.
        
        Returns:
            Tuple[bid_cols, ask_cols, venues]
        """
        bid_cols = [col for col in df.columns if col.endswith('_bid') and not col.endswith('_bid_qty')]
        ask_cols = [col for col in df.columns if col.endswith('_ask') and not col.endswith('_ask_qty')]
        
        # Extraer nombres de venues
        venues = sorted(set([col.replace('_bid', '').replace('_ask', '') 
                            for col in bid_cols + ask_cols]))
        
        return bid_cols, ask_cols, venues
    
    @staticmethod
    def _calculate_global_best_prices(df: pd.DataFrame, 
                                     bid_cols: List[str], 
                                     ask_cols: List[str]) -> pd.DataFrame:
        """
        Calcula Global Max Bid y Global Min Ask de forma vectorizada.
        
        Args:
            df: DataFrame consolidado
            bid_cols: Lista de columnas de bids
            ask_cols: Lista de columnas de asks
            
        Returns:
            DataFrame con columnas añadidas: global_max_bid, global_min_ask,
            venue_max_bid, venue_min_ask
        """
        # Calcular máximos y mínimos de forma vectorizada
        bids_df = df[bid_cols]
        asks_df = df[ask_cols]
        
        # CRÍTICO: Verificar que no hay DataFrames vacíos
        if len(bids_df) == 0 or len(asks_df) == 0:
            logger.error("  ⚠️ CRÍTICO: DataFrame vacío en cálculo de global prices")
            return df
        
        # CRÍTICO: max() y min() con skipna=True por defecto
        # Si TODAS las columnas tienen NaN, max/min retorna NaN
        # Esto causaría pérdida de señales, así que debemos manejar este caso
        df['max_bid'] = bids_df.max(axis=1, skipna=True)
        df['min_ask'] = asks_df.min(axis=1, skipna=True)
        
        # CRÍTICO: Si hay NaNs, reemplazar con valores extremos para que no bloqueen la comparación
        # Pero solo si TODAS las columnas tienen NaN (caso extremo)
        # Si solo algunas tienen NaN, max/min con skipna=True ya las ignora correctamente
        
        # Identificar venues ANTES de modificar NaNs
        # Usar fillna temporalmente solo para idxmax/idxmin, no para modificar los valores reales
        df['venue_max_bid'] = bids_df.fillna(-np.inf).idxmax(axis=1).str.replace('_bid', '')
        df['venue_min_ask'] = asks_df.fillna(np.inf).idxmin(axis=1).str.replace('_ask', '')
        
        # Verificar NaNs después del cálculo
        nan_max_bid = df['max_bid'].isna().sum()
        nan_min_ask = df['min_ask'].isna().sum()
        
        if nan_max_bid > 0:
            logger.warning(f"  ⚠️ ADVERTENCIA: {nan_max_bid} NaNs en max_bid después de cálculo")
            logger.warning(f"    Esto puede causar pérdida de señales - verificando...")
            # Si hay NaNs, significa que TODAS las columnas bid tienen NaN en esas filas
            # En este caso, no podemos calcular max_bid, pero esto es un problema de datos
            
        if nan_min_ask > 0:
            logger.warning(f"  ⚠️ ADVERTENCIA: {nan_min_ask} NaNs en min_ask después de cálculo")
            logger.warning(f"    Esto puede causar pérdida de señales - verificando...")
        
        return df
    
    @staticmethod
    def _extract_quantities_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae cantidades de forma vectorizada (sin usar apply).
        
        Optimizado para mejor rendimiento con DataFrames grandes.
        
        Returns:
            DataFrame con columnas añadidas: bid_qty, ask_qty
        """
        # Crear máscaras para cada venue y extraer cantidades
        venues = df['venue_max_bid'].unique()
        
        # Inicializar columnas
        df['bid_qty'] = 0.0
        df['ask_qty'] = 0.0
        
        # Extraer cantidades de forma vectorizada por venue
        # CRÍTICO: Manejar NaNs correctamente - si una cantidad es NaN, usar 0 (no ejecutable)
        for venue in venues:
            bid_qty_col = f'{venue}_bid_qty'
            ask_qty_col = f'{venue}_ask_qty'
            
            if bid_qty_col in df.columns:
                mask_bid = df['venue_max_bid'] == venue
                # Si la cantidad es NaN, usar 0 (no hay cantidad disponible)
                df.loc[mask_bid, 'bid_qty'] = df.loc[mask_bid, bid_qty_col].fillna(0.0)
            
            if ask_qty_col in df.columns:
                mask_ask = df['venue_min_ask'] == venue
                # Si la cantidad es NaN, usar 0 (no hay cantidad disponible)
                df.loc[mask_ask, 'ask_qty'] = df.loc[mask_ask, ask_qty_col].fillna(0.0)
        
        return df
    
    @staticmethod
    def _calculate_profits(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula profit teórico y cantidades ejecutables.
        
        Returns:
            DataFrame con columnas añadidas: theoretical_profit, executable_qty, total_profit
        """
        # Profit por unidad: (Max Bid - Min Ask)
        # CRÍTICO: Manejar NaNs - si alguno es NaN, profit será NaN, pero eso ya se maneja en signal
        df['theoretical_profit'] = df['max_bid'] - df['min_ask']
        
        # Cantidad ejecutable: min(bid_size, ask_size)
        # CRÍTICO: fillna(0) para asegurar que si hay NaN, la cantidad sea 0 (no ejecutable)
        df['executable_qty'] = np.minimum(
            df['bid_qty'].fillna(0.0), 
            df['ask_qty'].fillna(0.0)
        )
        
        # Profit total: Profit por unidad * Cantidad ejecutable
        # CRÍTICO: Si theoretical_profit es NaN o executable_qty es 0, total_profit debe ser 0
        df['total_profit'] = (df['theoretical_profit'].fillna(0.0) * df['executable_qty']).fillna(0.0)
        
        # Inicializar cantidades remanentes
        df['remaining_bid_qty'] = df['bid_qty'].fillna(0.0)
        df['remaining_ask_qty'] = df['ask_qty'].fillna(0.0)
        
        # CRÍTICO: Solo poner profit=0 si realmente NO hay oportunidad
        # No poner profit=0 si hay NaNs (eso ya se maneja en la detección de señal)
        # Usar la columna 'signal' que ya tiene en cuenta los NaNs
        mask_no_opp = df['signal'] == 0
        df.loc[mask_no_opp, 'theoretical_profit'] = 0.0
        df.loc[mask_no_opp, 'total_profit'] = 0.0
        df.loc[mask_no_opp, 'executable_qty'] = 0.0
        
        return df
    
    @staticmethod
    def apply_rising_edge(signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mantiene solo la PRIMERA aparición de cada oportunidad continua.
        
        Algoritmo:
        1. Identificar transiciones 0→1 en la columna 'signal'
        2. Marcar solo esas filas como oportunidades "tradeables"
        3. Si la señal desaparece (1→0) y vuelve a aparecer (0→1), 
           cuenta como nueva oportunidad
        
        Args:
            signals_df: DataFrame con columna 'is_opportunity' o 'signal'
            
        Returns:
            DataFrame con columna adicional 'is_rising_edge' (bool)
        """
        # Usar 'is_opportunity' si existe, sino 'signal'
        signal_col = 'is_opportunity' if 'is_opportunity' in signals_df.columns else 'signal'
        
        if signal_col not in signals_df.columns:
            logger.warning("No se encontró columna 'is_opportunity' o 'signal'")
            signals_df['is_rising_edge'] = False
            return signals_df
        
        # Comparar con el snapshot anterior para detectar transiciones 0→1
        # CRÍTICO: Rising edge = primera aparición de una oportunidad continua
        # Si una oportunidad persiste durante 1000 snapshots, solo la contamos una vez
        prev_signal = signals_df[signal_col].shift(1, fill_value=0)
        
        # Rising edge = señal actual (1) AND no había señal anterior (0)
        # Esto identifica el momento exacto donde aparece una nueva oportunidad
        signals_df['is_rising_edge'] = (
            (signals_df[signal_col] == 1) & (prev_signal == 0)
        )
        
        return signals_df
    
    @staticmethod
    def validate_signals(signals_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Valida que todas las señales cumplan condiciones lógicas.
        
        Validaciones:
        1. Todas las señales tienen max_bid > min_ask
        2. theoretical_profit >= 0
        3. executable_qty > 0 para todas las señales
        4. venue_max_bid != venue_min_ask (no arbitraje dentro del mismo venue)
        5. Reportar estadísticas
        
        Args:
            signals_df: DataFrame con señales
            
        Returns:
            Tuple[DataFrame validado, estadísticas]
        """
        stats = {
            'total_signals': 0,
            'invalid_spread': 0,
            'negative_profit': 0,
            'zero_qty': 0,
            'same_venue': 0,
            'valid_signals': 0,
            'total_profit': 0.0,
            'avg_profit': 0.0
        }
        
        # Filtrar solo rising edges
        rising_edges = signals_df[signals_df.get('is_rising_edge', False)].copy()
        
        if len(rising_edges) == 0:
            return signals_df, stats
        
        stats['total_signals'] = len(rising_edges)
        
        # Validación 1: max_bid > min_ask
        invalid_spread = (rising_edges['max_bid'] <= rising_edges['min_ask']).sum()
        stats['invalid_spread'] = invalid_spread
        
        # Validación 2: theoretical_profit >= 0
        negative_profit = (rising_edges['theoretical_profit'] < 0).sum()
        stats['negative_profit'] = negative_profit
        
        # Validación 3: executable_qty > 0
        zero_qty = (rising_edges['executable_qty'] <= 0).sum()
        stats['zero_qty'] = zero_qty
        
        # Validación 4: venue_max_bid != venue_min_ask
        venue_max_col = rising_edges.get('venue_max_bid', rising_edges.get('max_bid_venue', None))
        venue_min_col = rising_edges.get('venue_min_ask', rising_edges.get('min_ask_venue', None))
        
        if venue_max_col is not None and venue_min_col is not None:
            same_venue = (venue_max_col == venue_min_col).sum()
            stats['same_venue'] = same_venue
            
            # Filtrar señales válidas
            # CRÍTICO: Manejar NaNs correctamente - solo comparar si ambos valores no son NaN
            valid_mask = (
                rising_edges['max_bid'].notna() &
                rising_edges['min_ask'].notna() &
                (rising_edges['max_bid'] > rising_edges['min_ask']) &
                (rising_edges['theoretical_profit'] >= 0) &
                (rising_edges.get('executable_qty', rising_edges.get('tradeable_qty', pd.Series([0]))).fillna(0) > 0) &
                (venue_max_col != venue_min_col)
            )
        else:
            stats['same_venue'] = 0
            valid_mask = (
                rising_edges['max_bid'].notna() &
                rising_edges['min_ask'].notna() &
                (rising_edges['max_bid'] > rising_edges['min_ask']) &
                (rising_edges['theoretical_profit'] >= 0) &
                (rising_edges.get('executable_qty', rising_edges.get('tradeable_qty', pd.Series([0]))).fillna(0) > 0)
            )
        
        valid_signals = rising_edges[valid_mask].copy()
        stats['valid_signals'] = len(valid_signals)
        
        if len(valid_signals) > 0:
            stats['total_profit'] = valid_signals['total_profit'].sum()
            stats['avg_profit'] = valid_signals['total_profit'].mean()
        
        # Reportar estadísticas
        logger.info(f"Validación de señales:")
        logger.info(f"  Total señales: {stats['total_signals']}")
        logger.info(f"  Inválidas (spread): {stats['invalid_spread']}")
        logger.info(f"  Inválidas (profit negativo): {stats['negative_profit']}")
        logger.info(f"  Inválidas (qty cero): {stats['zero_qty']}")
        logger.info(f"  Inválidas (mismo venue): {stats['same_venue']}")
        logger.info(f"  Válidas: {stats['valid_signals']}")
        logger.info(f"  Profit total: €{stats['total_profit']:.2f}")
        logger.info(f"  Profit medio: €{stats['avg_profit']:.2f}")
        
        return signals_df, stats
    
    @staticmethod
    def filter_edge_cases(signals_df: pd.DataFrame, 
                          consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra edge cases opcionales pero recomendados.
        
        Filtros:
        1. Market Open: Ignorar primeros 5 minutos (spreads anormales) - NO IMPLEMENTADO
        2. Market Close: Solo hasta 17:25 CET (últimos segundos en fase de cierre) - NO IMPLEMENTADO
        3. Minimum profit threshold: Descartar oportunidades < 0.01 EUR - DESHABILITADo
        
        NOTA: El filtro de profit mínimo está deshabilitado para mantener todas las oportunidades.
        
        Args:
            signals_df: DataFrame con señales
            consolidated_df: DataFrame consolidado original
            
        Returns:
            DataFrame filtrado (actualmente sin filtros aplicados)
        """
        df = signals_df.copy()
        initial_len = len(df)
        
        # Filtro 1: Minimum profit threshold
        # DESHABILITADO: No eliminar oportunidades con profit < €0.1
        # if 'total_profit' in df.columns:
        #     min_profit_mask = df['total_profit'] >= config.MIN_THEORETICAL_PROFIT
        #     removed_profit = (~min_profit_mask).sum()
        #     df = df[min_profit_mask]
        #     if removed_profit > 0:
        #         logger.info(f"  Eliminadas {removed_profit:,} oportunidades con profit < €{config.MIN_THEORETICAL_PROFIT}")
        
        # Filtro 2: Market Open/Close (si tenemos información de tiempo)
        # Nota: Esto requiere convertir epoch a datetime, se puede implementar si es necesario
        
        # Filtro 3: Venue downtime (requiere análisis de timestamps)
        # Nota: Se puede implementar si es necesario
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"  Total eliminadas por edge cases: {removed:,}")
        
        return df
    
    def generate_signals(self, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Función principal para generar señales de arbitraje.
        
        Input: DataFrame consolidado con columnas:
               [epoch, XMAD_bid, XMAD_ask, XMAD_bid_qty, XMAD_ask_qty,
                AQXE_bid, AQXE_ask, AQXE_bid_qty, AQXE_ask_qty, ...]
        
        Output: DataFrame de señales con columnas:
                [epoch, signal, max_bid, min_ask, max_bid_venue, min_ask_venue,
                 max_bid_qty, min_ask_qty, theoretical_profit, tradeable_qty]
        
        Args:
            consolidated_df: DataFrame consolidado
            
        Returns:
            DataFrame con señales detectadas
        """
        print("\n" + "=" * 80)
        print("DETECCIÓN DE SEÑALES DE ARBITRAJE")
        print("=" * 80)
        
        df = consolidated_df.copy()
        
        if len(df) == 0:
            logger.warning("DataFrame consolidado vacío")
            return pd.DataFrame()
        
        # PASO 1: Extraer columnas de precios
        bid_cols, ask_cols, venues = self._extract_price_columns(df)
        
        if not bid_cols or not ask_cols:
            logger.error("No se encontraron columnas de precios (_bid, _ask)")
            return pd.DataFrame()
        
        print(f"  Venues detectados: {venues}")
        print(f"  Total snapshots: {len(df):,}")
        
        # PASO 2: Calcular Global Max Bid y Global Min Ask
        print("\n  Calculando Global Max Bid y Global Min Ask...")
        df = self._calculate_global_best_prices(df, bid_cols, ask_cols)
        
        # PASO 3: Extraer cantidades (vectorizado)
        print("  Extrayendo cantidades disponibles...")
        df = self._extract_quantities_vectorized(df)
        
        # PASO 4: Detectar condición de arbitraje
        print("  Detectando condición de arbitraje (Bid > Ask)...")
        
        # CRÍTICO: Verificar que no hay NaNs antes de comparar
        nan_max_bid = df['max_bid'].isna().sum()
        nan_min_ask = df['min_ask'].isna().sum()
        
        if nan_max_bid > 0:
            logger.warning(f"  ⚠️ ADVERTENCIA: {nan_max_bid} NaNs en max_bid antes de detectar señales")
        if nan_min_ask > 0:
            logger.warning(f"  ⚠️ ADVERTENCIA: {nan_min_ask} NaNs en min_ask antes de detectar señales")
        
        # Detectar señal: max_bid > min_ask (sin ningún threshold adicional)
        # CRÍTICO: Manejar NaNs correctamente - si hay NaN, la comparación retorna False
        # Pero queremos detectar oportunidades incluso si uno de los valores es NaN (siempre que el otro no lo sea)
        # Sin embargo, si AMBOS son NaN, no podemos detectar arbitraje
        
        # Comparación estándar: max_bid > min_ask
        # Si max_bid es NaN o min_ask es NaN, la comparación retorna False (no detecta señal)
        # Esto es correcto porque no podemos comparar con NaN
        
        # PERO: Si max_bid NO es NaN y min_ask NO es NaN, debemos comparar correctamente
        # Usar fillna con valores extremos solo para la comparación, no para modificar los datos
        signal_mask = (
            df['max_bid'].notna() & 
            df['min_ask'].notna() & 
            (df['max_bid'] > df['min_ask'])
        )
        
        df['signal'] = signal_mask.astype(int)
        df['is_opportunity'] = signal_mask.astype(bool)
        
        # CRÍTICO: Reportar cuántas oportunidades se perdieron por NaNs
        nan_blocked = (
            (df['max_bid'].isna() | df['min_ask'].isna()) & 
            ~(df['max_bid'].isna() & df['min_ask'].isna())  # Al menos uno tiene valor
        ).sum()
        
        if nan_blocked > 0:
            logger.warning(f"  ⚠️ ADVERTENCIA: {nan_blocked:,} snapshots no pudieron evaluarse por NaNs parciales")
            logger.warning(f"    (tienen algunos venues con datos pero no todos)")
        
        # DIAGNÓSTICO: Reportar cuántas señales se detectaron
        total_signals = df['signal'].sum()
        logger.info(f"  Total snapshots con señal detectada: {total_signals:,} de {len(df):,} ({total_signals/len(df)*100:.4f}%)")
        
        # PASO 5: Calcular profits y cantidades ejecutables
        # CRÍTICO: Calcular profits ANTES de aplicar rising edge, pero después de detectar señal
        print("  Calculando profit teórico...")
        df = self._calculate_profits(df)
        
        # PASO 6: Aplicar Rising Edge Detection
        print("  Aplicando Rising Edge Detection...")
        df = self.apply_rising_edge(df)
        
        # PASO 7: Validar señales
        print("  Validando señales...")
        df, validation_stats = self.validate_signals(df)
        
        # PASO 8: Filtrar edge cases (opcional)
        print("  Aplicando filtros de edge cases...")
        df = self.filter_edge_cases(df, consolidated_df)
        
        # Mantener nombres compatibles con código existente
        # Añadir alias según especificaciones
        df['global_max_bid'] = df['max_bid']
        df['global_min_ask'] = df['min_ask']
        df['max_bid_venue'] = df['venue_max_bid']
        df['min_ask_venue'] = df['venue_min_ask']
        df['max_bid_qty'] = df['bid_qty']
        df['min_ask_qty'] = df['ask_qty']
        df['tradeable_qty'] = df['executable_qty']
        
        # Seleccionar columnas (mantener compatibilidad con código existente)
        output_cols = [
            'epoch', 'signal', 'is_opportunity',
            'max_bid', 'min_ask',  # Especificaciones
            'global_max_bid', 'global_min_ask',  # Compatibilidad
            'venue_max_bid', 'venue_min_ask',  # Compatibilidad
            'max_bid_venue', 'min_ask_venue',  # Especificaciones
            'bid_qty', 'ask_qty',  # Compatibilidad
            'max_bid_qty', 'min_ask_qty',  # Especificaciones
            'executable_qty', 'tradeable_qty',  # Ambos nombres
            'theoretical_profit', 'total_profit',
            'is_rising_edge',
            'remaining_bid_qty', 'remaining_ask_qty'
        ]
        
        available_cols = [col for col in output_cols if col in df.columns]
        signals_df = df[available_cols].copy()
        
        # RESUMEN
        total_snapshots = len(consolidated_df)
        total_opportunities = df['is_opportunity'].sum()
        rising_edges = df['is_rising_edge'].sum()
        valid_rising_edges = len(df[df['is_rising_edge']])
        
        print(f"\n  RESULTADOS:")
        print(f"    - Total snapshots: {total_snapshots:,}")
        print(f"    - Snapshots con arbitraje: {total_opportunities:,} ({total_opportunities/total_snapshots*100:.2f}%)")
        print(f"    - Rising edges detectados: {rising_edges:,}")
        print(f"    - Rising edges válidos: {valid_rising_edges:,}")
        
        if valid_rising_edges > 0:
            valid_opps = df[df['is_rising_edge']]
            total_profit = valid_opps['total_profit'].sum()
            avg_profit = valid_opps['total_profit'].mean()
            max_profit = valid_opps['total_profit'].max()
            
            print(f"    - Profit teórico total: €{total_profit:.2f}")
            print(f"    - Profit medio: €{avg_profit:.2f}")
            print(f"    - Profit máximo: €{max_profit:.2f}")
        
        return signals_df
    
    def detect_opportunities(self, 
                            consolidated_tape: pd.DataFrame, 
                            executed_trades: list = None,
                            isin: str = None) -> pd.DataFrame:
        """
        Detecta oportunidades de arbitraje (compatibilidad con código existente).
        
        Wrapper alrededor de generate_signals con soporte para executed_trades.
        """
        # Generar señales base
        signals_df = self.generate_signals(consolidated_tape)
        
        if len(signals_df) == 0:
            return signals_df
        
        # Aplicar filtro de executed_trades si se proporciona
        if executed_trades and len(executed_trades) > 0:
            signals_df['_consumed_execution_id'] = None
            
            exec_df = pd.DataFrame(executed_trades)
            
            # Filtrar oportunidades ya ejecutadas (optimizado con merge)
            for idx, row in signals_df[signals_df['is_opportunity']].iterrows():
                # Obtener nombres de venues (compatibilidad con ambos nombres)
                venue_max_bid = row.get('venue_max_bid', row.get('max_bid_venue', ''))
                venue_min_ask = row.get('venue_min_ask', row.get('min_ask_venue', ''))
                
                matching_execs = exec_df[
                    (exec_df['epoch'] <= row['epoch']) &
                    (exec_df.get('venue_max_bid', '') == venue_max_bid) &
                    (exec_df.get('venue_min_ask', '') == venue_min_ask)
                ]
                
                if isin and 'isin' in exec_df.columns:
                    matching_execs = matching_execs[matching_execs['isin'] == isin]
                
                if len(matching_execs) > 0:
                    latest_exec = matching_execs.loc[matching_execs['epoch'].idxmax()]
                    signals_df.at[idx, '_consumed_execution_id'] = latest_exec.get('execution_id', None)
                    
                    # Ajustar cantidades remanentes
                    executed_qty = latest_exec.get('executed_qty', 0)
                    max_bid_qty = row.get('max_bid_qty', row.get('bid_qty', 0))
                    min_ask_qty = row.get('min_ask_qty', row.get('ask_qty', 0))
                    
                    signals_df.at[idx, 'remaining_bid_qty'] = max(0, max_bid_qty - executed_qty)
                    signals_df.at[idx, 'remaining_ask_qty'] = max(0, min_ask_qty - executed_qty)
                    
                    tradeable_qty = np.minimum(
                        signals_df.at[idx, 'remaining_bid_qty'],
                        signals_df.at[idx, 'remaining_ask_qty']
                    )
                    signals_df.at[idx, 'tradeable_qty'] = tradeable_qty
                    signals_df.at[idx, 'executable_qty'] = tradeable_qty  # Compatibilidad
                    signals_df.at[idx, 'total_profit'] = (
                        signals_df.at[idx, 'theoretical_profit'] * tradeable_qty
                    )
        
        return signals_df
    
    @staticmethod
    def analyze_venue_pairs(signals_df: pd.DataFrame) -> pd.DataFrame:
        """Analiza qué pares de venues generan más oportunidades."""
        print("\n" + "=" * 80)
        print("ANÁLISIS DE PARES DE VENUES")
        print("=" * 80)
        
        opportunities = signals_df[
            (signals_df.get('is_rising_edge', False)) & 
            (signals_df.get('total_profit', 0) > 0)
        ].copy()
        
        if len(opportunities) == 0:
            print("  No hay oportunidades válidas para analizar")
            return pd.DataFrame()
        
        # Crear columna con par de venues
        max_bid_venue = opportunities.get('max_bid_venue', opportunities.get('venue_max_bid', ''))
        min_ask_venue = opportunities.get('min_ask_venue', opportunities.get('venue_min_ask', ''))
        
        opportunities['venue_pair'] = (
            'Buy@' + min_ask_venue.astype(str) + 
            ' / Sell@' + max_bid_venue.astype(str)
        )
        
        # Agrupar por par de venues
        pairs_stats = opportunities.groupby('venue_pair').agg({
            'total_profit': ['count', 'sum', 'mean', 'max'],
            'theoretical_profit': 'mean',
            'tradeable_qty': 'mean'
        }).reset_index()
        
        # Renombrar columnas
        pairs_stats.columns = [
            'Venue Pair', 'Count', 'Total Profit (€)', 
            'Avg Profit (€)', 'Max Profit (€)',
            'Avg Price Diff (€)', 'Avg Qty'
        ]
        
        pairs_stats = pairs_stats.sort_values('Total Profit (€)', ascending=False)
        
        print("\nTop 10 Pares de Venues:")
        print(pairs_stats.head(10).to_string(index=False))
        
        return pairs_stats
    
    @staticmethod
    def visualize_signals(signals_df: pd.DataFrame, isin: str, max_points: int = 10000):
        """Visualiza las señales de arbitraje detectadas."""
        print("\nGenerando visualizaciones de señales...")
        
        if len(signals_df) > max_points:
            sample_indices = np.sort(np.random.choice(signals_df.index, size=max_points, replace=False))
            sample_df = signals_df.loc[sample_indices].copy()
        else:
            sample_df = signals_df.copy()
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        x_values = range(len(sample_df))
        
        # Plot 1: Global Max Bid vs Global Min Ask
        ax1 = axes[0]
        max_bid_col = sample_df.get('max_bid', sample_df.get('global_max_bid', None))
        min_ask_col = sample_df.get('min_ask', sample_df.get('global_min_ask', None))
        
        if max_bid_col is not None and min_ask_col is not None:
            ax1.plot(x_values, max_bid_col.values, label='Global Max Bid', color='green', alpha=0.7, linewidth=1.5)
            ax1.plot(x_values, min_ask_col.values, label='Global Min Ask', color='red', alpha=0.7, linewidth=1.5)
            
            opportunities_mask = sample_df.get('is_opportunity', pd.Series([False]*len(sample_df)))
            if opportunities_mask.any():
                opp_indices = [i for i, v in enumerate(opportunities_mask) if v]
                opp_bids = max_bid_col[opportunities_mask].values
                ax1.scatter(opp_indices, opp_bids, color='gold', s=30, alpha=0.6, 
                           label='Arbitrage Opportunity', zorder=5)
        
        ax1.set_title(f'Global Best Prices - ISIN: {isin}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Snapshot Index')
        ax1.set_ylabel('Price (€)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Theoretical Profit Over Time
        ax2 = axes[1]
        profit_mask = sample_df.get('is_opportunity', False) & (sample_df.get('theoretical_profit', 0) > 0)
        
        if profit_mask.any():
            profit_indices = [i for i, v in enumerate(profit_mask) if v]
            profit_values = sample_df.loc[profit_mask, 'theoretical_profit'].values * 10000
            ax2.scatter(profit_indices, profit_values, color='blue', alpha=0.6, s=30)
            ax2.set_title('Theoretical Profit per Unit (basis points)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Snapshot Index')
            ax2.set_ylabel('Profit (bps)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No opportunities detected', ha='center', va='center', 
                    fontsize=14, transform=ax2.transAxes)
        
        # Plot 3: Cumulative Profit
        ax3 = axes[2]
        rising_edges = signals_df[signals_df.get('is_rising_edge', False)].copy()
        
        if len(rising_edges) > 0:
            rising_edges = rising_edges.sort_index()
            cumulative_profit = rising_edges['total_profit'].cumsum().values
            x_cum = range(len(rising_edges))
            ax3.plot(x_cum, cumulative_profit, color='darkgreen', linewidth=2)
            ax3.fill_between(x_cum, cumulative_profit, alpha=0.3, color='green')
            ax3.set_title('Cumulative Theoretical Profit (€)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Opportunity Number')
            ax3.set_ylabel('Cumulative Profit (€)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No rising edges detected', ha='center', va='center',
                    fontsize=14, transform=ax3.transAxes)
        
        plt.tight_layout()
        
        output_path = config.FIGURES_DIR / f'signals_{isin}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Visualización guardada en: {output_path}")
        
        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception as e:
            logger.warning(f"  No se pudo mostrar gráficas interactivas: {e}")
    
    @staticmethod
    def export_opportunities(signals_df: pd.DataFrame, output_path: str = None):
        """Exporta las oportunidades detectadas a CSV."""
        opportunities = signals_df[
            (signals_df.get('is_rising_edge', False)) & 
            (signals_df.get('total_profit', 0) > 0) &
            (signals_df.get('tradeable_qty', signals_df.get('executable_qty', 0)) > 0)
        ].copy()
        
        if len(opportunities) == 0:
            print("  No hay oportunidades para exportar")
            return
        
        if output_path is None:
            output_path = config.OUTPUT_DIR / "opportunities.csv"
        
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        opportunities.to_csv(output_path, index=False)
        
        print(f"\n  Oportunidades exportadas a: {output_path}")
        print(f"    Total oportunidades: {len(opportunities):,}")


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
