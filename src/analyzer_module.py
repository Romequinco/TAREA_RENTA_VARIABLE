"""
FALTA AÑADIR AL MAIN SCRIPT.

================================================================================
analyzer.py - Módulo de Análisis y Métricas de Performance
================================================================================

OBJETIVO:
Generar el análisis final y métricas de performance del sistema de arbitraje.

MÉTRICAS CLAVE:
1. Profit total y medio por oportunidad
2. Tasa de éxito (% oportunidades profitable después de latencia)
3. Distribución temporal de oportunidades
4. Análisis por par de venues
5. Estadísticas de spreads y precios
6. ROI estimado considerando costes

OUTPUTS:
- Tablas resumen
- Métricas agregadas
- Estadísticas descriptivas
- Datos para visualizaciones

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
    """
    
    def __init__(self):
        """
        Inicializa el analizador.
        """
        logger.info("ArbitrageAnalyzer initialized")
    
    def analyze_opportunities(self, 
                            signals_df: pd.DataFrame,
                            exec_df: pd.DataFrame = None) -> Dict:
        """
        Análisis completo de oportunidades detectadas.
        
        Args:
            signals_df: DataFrame de signal_generator.detect_opportunities()
            exec_df: DataFrame de latency_simulator.simulate_execution() (opcional)
        
        Returns:
            Dict con todas las métricas calculadas
        """
        
        print("\n" + "=" * 80)
        print("ANÁLISIS DE OPORTUNIDADES DE ARBITRAJE")
        print("=" * 80)
        
        metrics = {}
        
        # ====================================================================
        # SECCIÓN 1: Métricas Básicas de Detección
        # ====================================================================
        print("\n[1] MÉTRICAS DE DETECCIÓN")
        print("-" * 80)
        
        # Total de snapshots analizados
        total_snapshots = len(signals_df)
        
        # Snapshots con condición de arbitraje
        total_arbitrage = signals_df['is_opportunity'].sum()
        
        # Rising edges (oportunidades únicas)
        total_rising_edges = signals_df['is_rising_edge'].sum()
        
        # Oportunidades válidas (con profit > threshold)
        valid_opportunities = signals_df[
            (signals_df['is_rising_edge']) & 
            (signals_df['total_profit'] > 0)
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
        
        print(f"  Total snapshots analizados: {total_snapshots:,}")
        print(f"  Snapshots con arbitraje: {total_arbitrage:,} ({metrics['detection']['arbitrage_rate_pct']:.4f}%)")
        print(f"  Rising edges detectados: {total_rising_edges:,}")
        print(f"  Oportunidades válidas: {total_valid:,} ({metrics['detection']['detection_rate_pct']:.4f}%)")
        
        # ====================================================================
        # SECCIÓN 2: Profit Teórico (sin latencia)
        # ====================================================================
        print("\n[2] PROFIT TEÓRICO (LATENCIA = 0)")
        print("-" * 80)
        
        if total_valid > 0:
            total_theoretical = valid_opportunities['total_profit'].sum()
            avg_theoretical = valid_opportunities['total_profit'].mean()
            median_theoretical = valid_opportunities['total_profit'].median()
            std_theoretical = valid_opportunities['total_profit'].std()
            max_theoretical = valid_opportunities['total_profit'].max()
            min_theoretical = valid_opportunities['total_profit'].min()
            
            metrics['theoretical_profit'] = {
                'total': total_theoretical,
                'mean': avg_theoretical,
                'median': median_theoretical,
                'std': std_theoretical,
                'max': max_theoretical,
                'min': min_theoretical
            }
            
            print(f"  Total profit teórico: €{total_theoretical:.2f}")
            print(f"  Profit medio: €{avg_theoretical:.4f}")
            print(f"  Profit mediano: €{median_theoretical:.4f}")
            print(f"  Desviación estándar: €{std_theoretical:.4f}")
            print(f"  Rango: €{min_theoretical:.4f} - €{max_theoretical:.4f}")
        else:
            metrics['theoretical_profit'] = None
            print("  No hay oportunidades válidas")
        
        # ====================================================================
        # SECCIÓN 3: Profit Real (con latencia) - Si disponible
        # ====================================================================
        if exec_df is not None and len(exec_df) > 0:
            print("\n[3] PROFIT REAL (CON LATENCIA)")
            print("-" * 80)
            
            # Filtrar solo profitable
            profitable_ops = exec_df[exec_df['profit_category'] == 'Profitable']
            
            total_real = exec_df['real_total_profit'].sum()
            profitable_count = len(profitable_ops)
            success_rate = (profitable_count / len(exec_df) * 100) if len(exec_df) > 0 else 0
            
            # Profit loss debido a latencia
            total_loss = exec_df['profit_loss_total'].sum()
            retention_rate = (total_real / total_theoretical * 100) if total_theoretical > 0 else 0
            
            metrics['real_profit'] = {
                'total': total_real,
                'mean': profitable_ops['real_total_profit'].mean() if len(profitable_ops) > 0 else 0,
                'median': profitable_ops['real_total_profit'].median() if len(profitable_ops) > 0 else 0,
                'profitable_count': profitable_count,
                'success_rate_pct': success_rate,
                'total_loss': total_loss,
                'retention_rate_pct': retention_rate
            }
            
            print(f"  Total profit real: €{total_real:.2f}")
            print(f"  Oportunidades profitable: {profitable_count:,} / {len(exec_df):,} ({success_rate:.1f}%)")
            print(f"  Pérdida por latencia: €{total_loss:.2f}")
            print(f"  Tasa de retención: {retention_rate:.1f}%")
            
            if len(profitable_ops) > 0:
                print(f"  Profit medio (solo profitable): €{metrics['real_profit']['mean']:.4f}")
        else:
            metrics['real_profit'] = None
        
        # ====================================================================
        # SECCIÓN 4: Análisis Temporal
        # ====================================================================
        print("\n[4] ANÁLISIS TEMPORAL")
        print("-" * 80)
        
        if total_valid > 0:
            # Convertir epoch a datetime
            valid_opportunities['datetime'] = pd.to_datetime(
                valid_opportunities['epoch'], 
                unit='ns'
            )
            
            # Duración total
            start_time = valid_opportunities['datetime'].min()
            end_time = valid_opportunities['datetime'].max()
            duration = end_time - start_time
            
            # Oportunidades por hora
            opportunities_per_hour = (total_valid / (duration.total_seconds() / 3600)) if duration.total_seconds() > 0 else 0
            
            # Distribución por minuto
            valid_opportunities['minute'] = valid_opportunities['datetime'].dt.floor('1min')
            ops_by_minute = valid_opportunities.groupby('minute').size()
            
            metrics['temporal'] = {
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration.total_seconds(),
                'opportunities_per_hour': opportunities_per_hour,
                'busiest_minute': ops_by_minute.idxmax() if len(ops_by_minute) > 0 else None,
                'max_ops_per_minute': ops_by_minute.max() if len(ops_by_minute) > 0 else 0
            }
            
            print(f"  Rango temporal: {start_time} - {end_time}")
            print(f"  Duración: {duration}")
            print(f"  Oportunidades por hora: {opportunities_per_hour:.2f}")
            print(f"  Minuto más activo: {metrics['temporal']['busiest_minute']}")
            print(f"  Max oportunidades/minuto: {metrics['temporal']['max_ops_per_minute']}")
        else:
            metrics['temporal'] = None
        
        # ====================================================================
        # SECCIÓN 5: Análisis por Pares de Venues
        # ====================================================================
        print("\n[5] ANÁLISIS POR PARES DE VENUES")
        print("-" * 80)
        
        if total_valid > 0:
            # Crear identificador de par
            valid_opportunities['venue_pair'] = (
                'Buy@' + valid_opportunities['venue_min_ask'] + 
                ' / Sell@' + valid_opportunities['venue_max_bid']
            )
            
            # Agrupar por par
            pairs_analysis = valid_opportunities.groupby('venue_pair').agg({
                'total_profit': ['count', 'sum', 'mean', 'max'],
                'theoretical_profit': 'mean',
                'executable_qty': 'mean'
            }).reset_index()
            
            pairs_analysis.columns = [
                'Venue Pair', 'Count', 'Total Profit', 'Avg Profit', 
                'Max Profit', 'Avg Price Diff', 'Avg Qty'
            ]
            
            pairs_analysis = pairs_analysis.sort_values('Total Profit', ascending=False)
            
            metrics['venue_pairs'] = pairs_analysis
            
            print(f"\n  Top 5 Pares de Venues:")
            print(pairs_analysis.head(5).to_string(index=False))
        else:
            metrics['venue_pairs'] = None
        
        # ====================================================================
        # SECCIÓN 6: Estadísticas de Spreads
        # ====================================================================
        print("\n[6] ESTADÍSTICAS DE SPREADS")
        print("-" * 80)
        
        if total_valid > 0:
            # Spread en basis points
            valid_opportunities['spread_bps'] = (
                valid_opportunities['theoretical_profit'] * 10000
            )
            
            spread_stats = {
                'mean_bps': valid_opportunities['spread_bps'].mean(),
                'median_bps': valid_opportunities['spread_bps'].median(),
                'std_bps': valid_opportunities['spread_bps'].std(),
                'min_bps': valid_opportunities['spread_bps'].min(),
                'max_bps': valid_opportunities['spread_bps'].max(),
                'q25_bps': valid_opportunities['spread_bps'].quantile(0.25),
                'q75_bps': valid_opportunities['spread_bps'].quantile(0.75)
            }
            
            metrics['spreads'] = spread_stats
            
            print(f"  Spread medio: {spread_stats['mean_bps']:.2f} bps")
            print(f"  Spread mediano: {spread_stats['median_bps']:.2f} bps")
            print(f"  Rango: {spread_stats['min_bps']:.2f} - {spread_stats['max_bps']:.2f} bps")
            print(f"  Percentiles (Q25, Q75): {spread_stats['q25_bps']:.2f}, {spread_stats['q75_bps']:.2f} bps")
        else:
            metrics['spreads'] = None
        
        print("\n" + "=" * 80)
        print("[ANÁLISIS COMPLETADO]")
        print("=" * 80)
        
        return metrics
    
    def estimate_roi(self,
                    metrics: Dict,
                    trading_costs_bps: float = 0.5,
                    capital_eur: float = 100000) -> Dict:
        """
        Estima ROI considerando costes de trading.
        
        COSTES TÍPICOS:
        - Comisiones: 0.1-0.5 bps por lado (0.2-1.0 bps total)
        - Slippage: 0.1-0.5 bps
        - Market impact: Variable según volumen
        
        Args:
            metrics: Dict de analyze_opportunities()
            trading_costs_bps: Costes totales de trading en basis points
            capital_eur: Capital disponible para trading
        
        Returns:
            Dict con estimaciones de ROI
        """
        
        print("\n" + "=" * 80)
        print("ESTIMACIÓN DE ROI")
        print("=" * 80)
        
        roi_metrics = {}
        
        # Verificar que hay datos
        if metrics.get('real_profit') is None:
            print("  [ADVERTENCIA] No hay datos de profit real para calcular ROI")
            return roi_metrics
        
        # Profit bruto (sin costes)
        gross_profit = metrics['real_profit']['total']
        
        # Calcular costes de trading
        # Costes = (spread en bps) * valor nominal de cada trade
        # Aproximación: usar profit total y asumir que cada oportunidad
        # involucra un valor nominal promedio
        
        total_ops = metrics['real_profit']['profitable_count']
        
        if total_ops == 0:
            print("  [ADVERTENCIA] No hay oportunidades profitable")
            return roi_metrics
        
        # Asumir valor nominal medio por operación
        # (esto debería calcularse con los datos reales de qty y precios)
        avg_notional_per_trade = capital_eur / 10  # Conservador: 10% del capital por trade
        
        # Costes totales = ops * notional * (costs_bps / 10000)
        trading_costs = total_ops * avg_notional_per_trade * (trading_costs_bps / 10000)
        
        # Profit neto
        net_profit = gross_profit - trading_costs
        
        # ROI = (net profit / capital) * 100
        roi_pct = (net_profit / capital_eur * 100) if capital_eur > 0 else 0
        
        # Duración del análisis
        duration_hours = metrics['temporal']['duration_seconds'] / 3600 if metrics.get('temporal') else 1
        
        # ROI anualizado (extrapolación)
        hours_per_year = 252 * 7  # 252 trading days * 7 hours/day
        roi_annualized = roi_pct * (hours_per_year / duration_hours) if duration_hours > 0 else 0
        
        roi_metrics = {
            'gross_profit': gross_profit,
            'trading_costs': trading_costs,
            'net_profit': net_profit,
            'capital': capital_eur,
            'roi_pct': roi_pct,
            'roi_annualized_pct': roi_annualized,
            'profitable_ops': total_ops,
            'avg_notional_per_trade': avg_notional_per_trade,
            'trading_costs_bps': trading_costs_bps
        }
        
        print(f"  Capital: €{capital_eur:,.0f}")
        print(f"  Profit bruto: €{gross_profit:.2f}")
        print(f"  Costes de trading ({trading_costs_bps} bps): €{trading_costs:.2f}")
        print(f"  Profit neto: €{net_profit:.2f}")
        print(f"  ROI: {roi_pct:.4f}%")
        print(f"  ROI anualizado (extrapolado): {roi_annualized:.2f}%")
        
        return roi_metrics
    
    def generate_summary_report(self,
                               metrics: Dict,
                               roi_metrics: Dict = None,
                               output_path: str = None) -> str:
        """
        Genera un reporte resumen en formato texto.
        
        Args:
            metrics: Dict de analyze_opportunities()
            roi_metrics: Dict de estimate_roi() (opcional)
            output_path: Path para guardar el reporte (opcional)
        
        Returns:
            String con el reporte completo
        """
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DE ANÁLISIS DE ARBITRAJE - MERCADOS FRAGMENTADOS EUROPEOS")
        report_lines.append("=" * 80)
        report_lines.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Sección 1: Detección
        if metrics.get('detection'):
            report_lines.append("[1] MÉTRICAS DE DETECCIÓN")
            report_lines.append("-" * 80)
            d = metrics['detection']
            report_lines.append(f"Total snapshots analizados: {d['total_snapshots']:,}")
            report_lines.append(f"Snapshots con arbitraje: {d['snapshots_with_arbitrage']:,} ({d['arbitrage_rate_pct']:.4f}%)")
            report_lines.append(f"Oportunidades válidas: {d['valid_opportunities']:,} ({d['detection_rate_pct']:.4f}%)")
            report_lines.append("")
        
        # Sección 2: Profit Teórico
        if metrics.get('theoretical_profit'):
            report_lines.append("[2] PROFIT TEÓRICO (LATENCIA = 0)")
            report_lines.append("-" * 80)
            tp = metrics['theoretical_profit']
            report_lines.append(f"Total: €{tp['total']:.2f}")
            report_lines.append(f"Medio: €{tp['mean']:.4f}")
            report_lines.append(f"Mediano: €{tp['median']:.4f}")
            report_lines.append(f"Rango: €{tp['min']:.4f} - €{tp['max']:.4f}")
            report_lines.append("")
        
        # Sección 3: Profit Real
        if metrics.get('real_profit'):
            report_lines.append("[3] PROFIT REAL (CON LATENCIA)")
            report_lines.append("-" * 80)
            rp = metrics['real_profit']
            report_lines.append(f"Total: €{rp['total']:.2f}")
            report_lines.append(f"Tasa de éxito: {rp['success_rate_pct']:.1f}%")
            report_lines.append(f"Pérdida por latencia: €{rp['total_loss']:.2f}")
            report_lines.append(f"Retención de profit: {rp['retention_rate_pct']:.1f}%")
            report_lines.append("")
        
        # Sección 4: Temporal
        if metrics.get('temporal'):
            report_lines.append("[4] ANÁLISIS TEMPORAL")
            report_lines.append("-" * 80)
            t = metrics['temporal']
            report_lines.append(f"Duración: {t['duration_seconds']:.0f} segundos")
            report_lines.append(f"Oportunidades/hora: {t['opportunities_per_hour']:.2f}")
            report_lines.append("")
        
        # Sección 5: ROI
        if roi_metrics:
            report_lines.append("[5] RETORNO DE INVERSIÓN")
            report_lines.append("-" * 80)
            report_lines.append(f"Capital: €{roi_metrics['capital']:,.0f}")
            report_lines.append(f"Profit neto: €{roi_metrics['net_profit']:.2f}")
            report_lines.append(f"ROI: {roi_metrics['roi_pct']:.4f}%")
            report_lines.append(f"ROI anualizado: {roi_metrics['roi_annualized_pct']:.2f}%")
            report_lines.append("")
        
        # Sección 6: Top Venue Pairs
        if metrics.get('venue_pairs') is not None:
            report_lines.append("[6] TOP PARES DE VENUES")
            report_lines.append("-" * 80)
            top5 = metrics['venue_pairs'].head(5)
            report_lines.append(top5.to_string(index=False))
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("[FIN DEL REPORTE]")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Guardar si se especifica path
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n  Reporte guardado en: {output_path}")
        
        return report
