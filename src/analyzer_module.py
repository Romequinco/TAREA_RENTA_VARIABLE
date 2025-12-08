"""
================================================================================
analyzer.py - Módulo de Análisis y Métricas de Performance (OPTIMIZADO)
================================================================================

OBJETIVO:
Generar métricas agregadas y estadísticas descriptivas que respondan las tres
preguntas del ejercicio:

1. ¿Existen oportunidades de arbitraje?
2. ¿Cuál es el profit teórico máximo?
3. ¿Cómo decae el profit con la latencia?

FUNCIONES PRINCIPALES:
- generate_isin_summary: Resumen por ISIN
- create_money_table: Tabla pivote con ISINs y latencias
- calculate_decay_curve: Curva de decay agregada
- identify_top_opportunities: Top oportunidades individuales

OPTIMIZACIONES APLICADAS:
- Agregaciones eficientes con pandas
- Validaciones y sanity checks integrados
- Formateo optimizado para presentación
- Compatibilidad con código existente

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from config_module import config

logger = logging.getLogger(__name__)


class ArbitrageAnalyzer:
    """
    Analiza resultados del sistema de arbitraje y genera métricas.
    
    Responde las tres preguntas clave:
    1. ¿Existen oportunidades de arbitraje?
    2. ¿Cuál es el profit teórico máximo?
    3. ¿Cómo decae el profit con la latencia?
    """
    
    def __init__(self):
        """Inicializa el analizador."""
        logger.info("ArbitrageAnalyzer initialized")
    
    @staticmethod
    def generate_isin_summary(signals_dict: Dict[str, pd.DataFrame],
                              latency_results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Genera resumen por ISIN con métricas clave.
        
        Input:
          - signals_dict: Dict[isin] -> DataFrame de señales
          - latency_results_dict: Dict[isin] -> DataFrame de resultados por latencia
        
        Output: DataFrame con columnas:
                [isin, ticker, total_opportunities, theoretical_profit_0lat,
                 actual_profit_100us, actual_profit_1ms, actual_profit_10ms,
                 actual_profit_100ms, best_venue_pair, avg_opportunity_duration]
        
        Args:
            signals_dict: Dict con señales por ISIN
            latency_results_dict: Dict con resultados de latencia por ISIN
            
        Returns:
            DataFrame con resumen por ISIN
        """
        print("\n" + "=" * 80)
        print("GENERANDO RESUMEN POR ISIN")
        print("=" * 80)
        
        summaries = []
        
        for isin, signals_df in signals_dict.items():
            if signals_df is None or len(signals_df) == 0:
                continue
            
            # Extraer ticker si está disponible
            ticker = signals_df.get('ticker', pd.Series([None])).iloc[0] if 'ticker' in signals_df.columns else None
            
            # Total oportunidades (rising edges)
            rising_edges = signals_df[signals_df.get('is_rising_edge', False)]
            total_opportunities = len(rising_edges)
            
            if total_opportunities == 0:
                continue
            
            # Profit teórico (latencia = 0)
            theoretical_profit_0lat = rising_edges['total_profit'].sum()
            
            # Obtener resultados de latencia para este ISIN
            latency_results = latency_results_dict.get(isin, pd.DataFrame())
            
            # Extraer profits por latencia específica (optimizado con dict lookup O(1))
            # Crear diccionario latencia -> profit para búsqueda rápida
            latency_profits = {}
            if len(latency_results) > 0:
                # Iterar una sola vez y crear mapa de latencias
                for _, row in latency_results.iterrows():
                    latency = row.get('latency_us', 0)
                    profit = row.get('total_actual_profit', 0.0)
                    latency_profits[latency] = profit
            
            # Extraer profits para latencias específicas (usando dict.get para evitar KeyError)
            actual_profit_100us = latency_profits.get(100, 0.0)      # 0.1ms
            actual_profit_1ms = latency_profits.get(1000, 0.0)       # 1ms
            actual_profit_10ms = latency_profits.get(10000, 0.0)    # 10ms
            actual_profit_100ms = latency_profits.get(100000, 0.0)  # 100ms
            
            # Mejor par de venues
            if 'venue_max_bid' in rising_edges.columns and 'venue_min_ask' in rising_edges.columns:
                venue_pairs = (
                    'Buy@' + rising_edges['venue_min_ask'].astype(str) + 
                    ' / Sell@' + rising_edges['venue_max_bid'].astype(str)
                )
                best_venue_pair = venue_pairs.value_counts().index[0] if len(venue_pairs) > 0 else 'N/A'
            else:
                best_venue_pair = 'N/A'
            
            # Duración promedio de oportunidades
            # Calcular duración estimada basada en la frecuencia de oportunidades
            if 'epoch' in rising_edges.columns and len(rising_edges) > 1:
                epochs = rising_edges['epoch'].sort_values()
                time_diffs = epochs.diff().dropna()
                avg_opportunity_duration = time_diffs.mean() / 1e9  # Convertir a segundos
            else:
                avg_opportunity_duration = 0.0
            
            summaries.append({
                'isin': isin,
                'ticker': ticker,
                'total_opportunities': total_opportunities,
                'theoretical_profit_0lat': theoretical_profit_0lat,
                'actual_profit_100us': actual_profit_100us,
                'actual_profit_1ms': actual_profit_1ms,
                'actual_profit_10ms': actual_profit_10ms,
                'actual_profit_100ms': actual_profit_100ms,
                'best_venue_pair': best_venue_pair,
                'avg_opportunity_duration': avg_opportunity_duration
            })
        
        summary_df = pd.DataFrame(summaries)
        
        if len(summary_df) > 0:
            print(f"\n  Resumen generado para {len(summary_df)} ISINs")
            print(summary_df.to_string(index=False))
        else:
            print("  No hay datos para generar resumen")
        
        return summary_df
    
    @staticmethod
    def create_money_table(latency_results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Tabla pivote con ISINs en filas y latencias en columnas.
        
        Formato:
                 0µs     100µs   1ms    10ms   100ms
        ISIN_A   1234€   1100€   890€   320€   45€
        ISIN_B   567€    520€    410€   120€   12€
        ...
        TOTAL    5678€   5123€   3456€  980€   123€
        
        Args:
            latency_results_dict: Dict[isin] -> DataFrame de resultados por latencia
            
        Returns:
            DataFrame formateado para presentación
        """
        print("\n" + "=" * 80)
        print("CREANDO MONEY TABLE")
        print("=" * 80)
        
        # Latencias estándar para columnas
        latency_columns = [0, 100, 500, 1000, 2000, 5000, 10000, 50000, 100000]
        latency_labels = ['0µs', '100µs', '500µs', '1ms', '2ms', '5ms', '10ms', '50ms', '100ms']
        
        money_data = []
        
        for isin, latency_df in latency_results_dict.items():
            if latency_df is None or len(latency_df) == 0:
                continue
            
            # Crear diccionario de latencia -> profit para lookup rápido
            latency_profit_map = dict(zip(
                latency_df['latency_us'],
                latency_df['total_actual_profit']
            ))
            
            row = {'ISIN': isin}
            
            for latency, label in zip(latency_columns, latency_labels):
                row[label] = latency_profit_map.get(latency, 0.0)
            
            money_data.append(row)
        
        if not money_data:
            print("  No hay datos para crear money table")
            return pd.DataFrame()
        
        money_df = pd.DataFrame(money_data)
        
        # Añadir fila TOTAL (calcular antes de formatear)
        total_row = {'ISIN': 'TOTAL'}
        for col in latency_labels:
            if col in money_df.columns:
                total_row[col] = money_df[col].sum()
        
        # Formatear valores como euros (mantener valores numéricos para TOTAL)
        money_df_formatted = money_df.copy()
        for col in latency_labels:
            if col in money_df_formatted.columns:
                money_df_formatted[col] = money_df_formatted[col].apply(
                    lambda x: f'€{x:,.2f}' if pd.notna(x) else '€0.00'
                )
        
        # Añadir fila TOTAL formateada
        total_row_formatted = {'ISIN': 'TOTAL'}
        for col in latency_labels:
            if col in total_row:
                total_row_formatted[col] = f'€{total_row[col]:,.2f}'
        
        money_df_formatted = pd.concat(
            [money_df_formatted, pd.DataFrame([total_row_formatted])], 
            ignore_index=True
        )
        
        print(f"\n  Money Table generada:")
        print(money_df_formatted.to_string(index=False))
        
        return money_df_formatted
    
    @staticmethod
    def calculate_decay_curve(latency_results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Agrega todos los ISINs para obtener la curva global de decay.
        
        Output: DataFrame con columnas:
                [latency_us, total_profit, profit_pct_of_theoretical,
                 opportunities_still_profitable]
        
        Args:
            latency_results_dict: Dict[isin] -> DataFrame de resultados por latencia
            
        Returns:
            DataFrame con curva de decay agregada
        """
        print("\n" + "=" * 80)
        print("CALCULANDO CURVA DE DECAY")
        print("=" * 80)
        
        # Agregar todos los resultados por latencia
        all_results = []
        
        for isin, latency_df in latency_results_dict.items():
            if latency_df is None or len(latency_df) == 0:
                continue
            
            all_results.append(latency_df)
        
        if not all_results:
            print("  No hay datos para calcular curva de decay")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Agrupar por latencia y agregar
        decay_curve = combined_df.groupby('latency_us').agg({
            'total_theoretical_profit': 'sum',
            'total_actual_profit': 'sum',
            'total_opportunities': 'sum',
            'profit_capture_rate': 'mean'  # Promedio ponderado sería mejor, pero promedio es más simple
        }).reset_index()
        
        # Calcular porcentaje del teórico
        decay_curve['profit_pct_of_theoretical'] = (
            (decay_curve['total_actual_profit'] / decay_curve['total_theoretical_profit'] * 100)
            .fillna(0.0)
        )
        
        # Renombrar columnas
        decay_curve = decay_curve.rename(columns={
            'total_actual_profit': 'total_profit',
            'total_opportunities': 'opportunities_still_profitable'
        })
        
        # Seleccionar columnas según especificaciones
        output_cols = [
            'latency_us', 'total_profit', 'profit_pct_of_theoretical',
            'opportunities_still_profitable'
        ]
        
        decay_curve = decay_curve[output_cols].copy()
        decay_curve = decay_curve.sort_values('latency_us')
        
        print(f"\n  Curva de decay generada:")
        print(decay_curve.to_string(index=False))
        
        return decay_curve
    
    @staticmethod
    def identify_top_opportunities(signals_dict: Dict[str, pd.DataFrame],
                                  n: int = 5) -> pd.DataFrame:
        """
        Identifica las N oportunidades individuales más rentables.
        
        Output: DataFrame con columnas:
                [isin, epoch, datetime, max_bid_venue, min_ask_venue,
                 spread_bps, theoretical_profit, tradeable_qty]
        
        SANITY CHECKS:
        - Spread en basis points (bps) debe ser razonable (<50 bps)
        - Cantidades deben ser realistas (100-10000 shares típicamente)
        - Venues deben ser diferentes
        - Timestamps deben estar en horario de trading (09:00-17:30 CET)
        
        Args:
            signals_dict: Dict[isin] -> DataFrame de señales
            n: Número de top oportunidades a identificar
            
        Returns:
            DataFrame con top oportunidades
        """
        print("\n" + "=" * 80)
        print(f"IDENTIFICANDO TOP {n} OPORTUNIDADES")
        print("=" * 80)
        
        all_opportunities = []
        
        for isin, signals_df in signals_dict.items():
            if signals_df is None or len(signals_df) == 0:
                continue
            
            # Filtrar solo rising edges válidos
            rising_edges = signals_df[
                (signals_df.get('is_rising_edge', False)) &
                (signals_df.get('total_profit', 0) > 0)
            ].copy()
            
            if len(rising_edges) == 0:
                continue
            
            # Añadir ISIN
            rising_edges['isin'] = isin
            
            # Convertir epoch a datetime
            rising_edges['datetime'] = pd.to_datetime(rising_edges['epoch'], unit='ns')
            
            # Extraer venues (compatibilidad con ambos nombres)
            rising_edges['max_bid_venue'] = rising_edges.get(
                'venue_max_bid', 
                rising_edges.get('max_bid_venue', '')
            )
            rising_edges['min_ask_venue'] = rising_edges.get(
                'venue_min_ask',
                rising_edges.get('min_ask_venue', '')
            )
            
            # Calcular spread en basis points
            theoretical_profit = rising_edges.get('theoretical_profit', 0.0)
            rising_edges['spread_bps'] = theoretical_profit * 10000
            
            # Obtener cantidad tradeable
            rising_edges['tradeable_qty'] = rising_edges.get(
                'tradeable_qty',
                rising_edges.get('executable_qty', 0)
            )
            
            # Seleccionar columnas relevantes
            cols = [
                'isin', 'epoch', 'datetime', 'max_bid_venue', 'min_ask_venue',
                'spread_bps', 'theoretical_profit', 'tradeable_qty', 'total_profit'
            ]
            
            available_cols = [col for col in cols if col in rising_edges.columns]
            opportunities_subset = rising_edges[available_cols].copy()
            
            all_opportunities.append(opportunities_subset)
        
        if not all_opportunities:
            print("  No hay oportunidades para analizar")
            return pd.DataFrame()
        
        combined_opps = pd.concat(all_opportunities, ignore_index=True)
        
        # SANITY CHECKS: Validaciones para filtrar oportunidades sospechosas o inválidas
        # Estas validaciones ayudan a identificar errores en los datos o oportunidades
        # que no son ejecutables en la práctica
        print(f"\n  Aplicando sanity checks...")
        
        # Check 1: Spread razonable (<50 bps)
        # Spreads muy grandes (>50 bps) pueden indicar errores en los datos o
        # situaciones de mercado anormales (halts, auctions, etc.)
        reasonable_spread = combined_opps['spread_bps'] < 50
        print(f"    Spreads razonables (<50 bps): {reasonable_spread.sum():,} / {len(combined_opps):,}")
        
        # Check 2: Cantidades realistas (100-10000 shares)
        # Cantidades muy pequeñas pueden ser errores, muy grandes pueden ser ilíquidas
        realistic_qty = (combined_opps['tradeable_qty'] >= 100) & (combined_opps['tradeable_qty'] <= 10000)
        print(f"    Cantidades realistas (100-10000): {realistic_qty.sum():,} / {len(combined_opps):,}")
        
        # Check 3: Venues diferentes
        # CRÍTICO: Arbitraje requiere comprar en un venue y vender en otro
        # Si ambos venues son iguales, no hay arbitraje real
        different_venues = combined_opps['max_bid_venue'] != combined_opps['min_ask_venue']
        print(f"    Venues diferentes: {different_venues.sum():,} / {len(combined_opps):,}")
        
        # Check 4: Horario de trading (09:00-17:30 CET)
        # Fuera del horario de trading continuo, los precios pueden ser de auctions
        # o pre-market/post-market, que no son ejecutables instantáneamente
        if 'datetime' in combined_opps.columns:
            combined_opps['hour'] = combined_opps['datetime'].dt.hour
            trading_hours = (combined_opps['hour'] >= 9) & (combined_opps['hour'] < 17)
            print(f"    En horario de trading (09:00-17:30): {trading_hours.sum():,} / {len(combined_opps):,}")
        
        # Combinar todos los checks en una máscara única
        valid_mask = reasonable_spread & realistic_qty & different_venues
        if 'datetime' in combined_opps.columns:
            valid_mask = valid_mask & trading_hours
        
        valid_opps = combined_opps[valid_mask].copy()
        
        # Ordenar por profit total y tomar top N
        top_opps = valid_opps.nlargest(n, 'total_profit')
        
        # Limpiar columnas temporales
        if 'hour' in top_opps.columns:
            top_opps = top_opps.drop('hour', axis=1)
        
        print(f"\n  Top {n} oportunidades identificadas:")
        print(top_opps.to_string(index=False))
        
        return top_opps
    
    def analyze_opportunities(self, 
                            signals_df: pd.DataFrame,
                            exec_df: pd.DataFrame = None) -> Dict:
        """
        Análisis completo de oportunidades detectadas.
        
        Compatibilidad con código existente.
        
        Args:
            signals_df: DataFrame de señales detectadas
            exec_df: DataFrame de ejecuciones (opcional)
        
        Returns:
            Dict con todas las métricas calculadas
        """
        print("\n" + "=" * 80)
        print("ANÁLISIS DE OPORTUNIDADES DE ARBITRAJE")
        print("=" * 80)
        
        metrics = {}
        
        # Métricas básicas
        total_snapshots = len(signals_df)
        total_arbitrage = signals_df.get('is_opportunity', pd.Series([False])).sum()
        total_rising_edges = signals_df.get('is_rising_edge', pd.Series([False])).sum()
        
        valid_opportunities = signals_df[
            (signals_df.get('is_rising_edge', False)) & 
            (signals_df.get('total_profit', 0) > 0)
        ]
        total_valid = len(valid_opportunities)
        
        metrics['detection'] = {
            'total_snapshots': total_snapshots,
            'snapshots_with_arbitrage': total_arbitrage,
            'arbitrage_rate_pct': (total_arbitrage / total_snapshots * 100) if total_snapshots > 0 else 0,
            'total_rising_edges': total_rising_edges,
            'valid_opportunities': total_valid,
            'detection_rate_pct': (total_valid / total_snapshots * 100) if total_snapshots > 0 else 0
        }
        
        print(f"  Total snapshots: {total_snapshots:,}")
        print(f"  Rising edges: {total_rising_edges:,}")
        print(f"  Oportunidades válidas: {total_valid:,}")
        
        # Profit teórico
        if total_valid > 0:
            total_theoretical = valid_opportunities['total_profit'].sum()
            metrics['theoretical_profit'] = {
                'total': total_theoretical,
                'mean': valid_opportunities['total_profit'].mean(),
                'max': valid_opportunities['total_profit'].max()
            }
            print(f"  Profit teórico total: €{total_theoretical:.2f}")
        else:
            metrics['theoretical_profit'] = None
        
        # Profit real (si disponible)
        if exec_df is not None and len(exec_df) > 0:
            total_real = exec_df.get('real_total_profit', pd.Series([0])).sum()
            profitable_count = (exec_df.get('profit_category', '') == 'Profitable').sum()
            
            metrics['real_profit'] = {
                'total': total_real,
                'profitable_count': profitable_count,
                'success_rate_pct': (profitable_count / len(exec_df) * 100) if len(exec_df) > 0 else 0
            }
            print(f"  Profit real: €{total_real:.2f}")
            print(f"  Oportunidades profitable: {profitable_count:,}")
        else:
            metrics['real_profit'] = None
        
        # Análisis temporal
        if total_valid > 0 and 'epoch' in valid_opportunities.columns:
            valid_opportunities['datetime'] = pd.to_datetime(valid_opportunities['epoch'], unit='ns')
            start_time = valid_opportunities['datetime'].min()
            end_time = valid_opportunities['datetime'].max()
            duration = end_time - start_time
            
            metrics['temporal'] = {
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration.total_seconds()
            }
        
        # Análisis por pares de venues
        if total_valid > 0:
            venue_max_col = valid_opportunities.get('venue_max_bid', valid_opportunities.get('max_bid_venue', ''))
            venue_min_col = valid_opportunities.get('venue_min_ask', valid_opportunities.get('min_ask_venue', ''))
            
            if venue_max_col is not None and venue_min_col is not None:
                valid_opportunities['venue_pair'] = (
                    'Buy@' + venue_min_col.astype(str) + 
                    ' / Sell@' + venue_max_col.astype(str)
                )
                
                pairs_analysis = valid_opportunities.groupby('venue_pair').agg({
                    'total_profit': ['count', 'sum', 'mean']
                }).reset_index()
                
                pairs_analysis.columns = ['Venue Pair', 'Count', 'Total Profit', 'Avg Profit']
                pairs_analysis = pairs_analysis.sort_values('Total Profit', ascending=False)
                
                metrics['venue_pairs'] = pairs_analysis
        
        return metrics
    
    def estimate_roi(self,
                    metrics: Dict,
                    trading_costs_bps: float = 0.5,
                    capital_eur: float = 100000) -> Dict:
        """
        Estima ROI considerando costes de trading.
        
        Compatibilidad con código existente.
        """
        print("\n" + "=" * 80)
        print("ESTIMACIÓN DE ROI")
        print("=" * 80)
        
        roi_metrics = {}
        
        if metrics.get('real_profit') is None:
            print("  [ADVERTENCIA] No hay datos de profit real para calcular ROI")
            return roi_metrics
        
        gross_profit = metrics['real_profit']['total']
        total_ops = metrics['real_profit'].get('profitable_count', 0)
        
        if total_ops == 0:
            return roi_metrics
        
        avg_notional_per_trade = capital_eur / 10
        trading_costs = total_ops * avg_notional_per_trade * (trading_costs_bps / 10000)
        net_profit = gross_profit - trading_costs
        roi_pct = (net_profit / capital_eur * 100) if capital_eur > 0 else 0
        
        duration_hours = metrics.get('temporal', {}).get('duration_seconds', 3600) / 3600
        hours_per_year = 252 * 7
        roi_annualized = roi_pct * (hours_per_year / duration_hours) if duration_hours > 0 else 0
        
        roi_metrics = {
            'gross_profit': gross_profit,
            'trading_costs': trading_costs,
            'net_profit': net_profit,
            'capital': capital_eur,
            'roi_pct': roi_pct,
            'roi_annualized_pct': roi_annualized
        }
        
        print(f"  ROI: {roi_pct:.4f}%")
        print(f"  ROI anualizado: {roi_annualized:.2f}%")
        
        return roi_metrics
    
    def generate_summary_report(self,
                               metrics: Dict,
                               roi_metrics: Dict = None,
                               output_path: str = None) -> str:
        """
        Genera un reporte resumen en formato texto.
        
        Compatibilidad con código existente.
        """
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DE ANÁLISIS DE ARBITRAJE")
        report_lines.append("=" * 80)
        report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if metrics.get('detection'):
            d = metrics['detection']
            report_lines.append(f"Oportunidades válidas: {d['valid_opportunities']:,}")
        
        if metrics.get('theoretical_profit'):
            tp = metrics['theoretical_profit']
            report_lines.append(f"Profit teórico: €{tp['total']:.2f}")
        
        if metrics.get('real_profit'):
            rp = metrics['real_profit']
            report_lines.append(f"Profit real: €{rp['total']:.2f}")
        
        if roi_metrics:
            report_lines.append(f"ROI: {roi_metrics['roi_pct']:.4f}%")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


# ============================================================================
# FUNCIONES WRAPPER ESTÁTICAS (según especificaciones)
# ============================================================================

def generate_isin_summary(signals_dict: Dict[str, pd.DataFrame],
                          latency_results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Genera resumen por ISIN con métricas clave.
    
    Wrapper estático según especificaciones.
    """
    return ArbitrageAnalyzer.generate_isin_summary(signals_dict, latency_results_dict)


def create_money_table(latency_results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Tabla pivote con ISINs en filas y latencias en columnas.
    
    Wrapper estático según especificaciones.
    """
    return ArbitrageAnalyzer.create_money_table(latency_results_dict)


def calculate_decay_curve(latency_results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Agrega todos los ISINs para obtener la curva global de decay.
    
    Wrapper estático según especificaciones.
    """
    return ArbitrageAnalyzer.calculate_decay_curve(latency_results_dict)


def identify_top_opportunities(signals_dict: Dict[str, pd.DataFrame],
                                n: int = 5) -> pd.DataFrame:
    """
    Identifica las N oportunidades individuales más rentables.
    
    Wrapper estático según especificaciones.
    """
    return ArbitrageAnalyzer.identify_top_opportunities(signals_dict, n)
