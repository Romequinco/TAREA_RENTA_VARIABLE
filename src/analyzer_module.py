"""
================================================================================
analyzer_module.py - Módulo de Análisis y Reportes
================================================================================

Funciones principales:
- create_money_table: Tabla pivot con profit por ISIN y latencia
- create_decay_chart: Gráficos de decay del profit con latencia
- identify_top_opportunities: Top 5 ISINs más rentables con detalles
- generate_summary_answers: Respuestas a las 3 preguntas clave
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from config_module import config

logger = logging.getLogger(__name__)


def create_money_table(money_table_data: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crea la Money Table (tabla pivot) con profit por ISIN y latencia.
    
    Args:
        money_table_data: Lista de diccionarios con estructura:
                         [{'ISIN': str, 'Latency_us': int, 'Profit_EUR': float}, ...]
        
    Returns:
        Tuple[pivot_table, summary_df]:
        - pivot_table: Tabla pivot con ISINs en filas y latencias en columnas
        - summary_df: Resumen agregado por latencia
    """
    if not money_table_data:
        print("No data available for Money Table.")
        return pd.DataFrame(), pd.DataFrame()
    
    money_df = pd.DataFrame(money_table_data)
    
    # Crear tabla pivot
    pivot = money_df.pivot_table(
        index='ISIN',
        columns='Latency_us',
        values='Profit_EUR',
        aggfunc='sum'
    )
    
    # Add TOTAL row
    pivot.loc['TOTAL'] = pivot.sum()
    
    print("MONEY TABLE: Total Realized Profit/Loss by ISIN and Latency")
    print("="*120)
    print(pivot.to_string())
    
    # Resumen por latencia
    print("\n" + "="*80)
    print("SUMMARY BY LATENCY (All ISINs Combined)")
    print("="*80)
    
    # Calculate profit at 0 latency for percentage calculation
    profit_at_zero = pivot.loc['TOTAL', 0] if 0 in pivot.columns else 0
    
    summary_df = pd.DataFrame({
        'Latency (µs)': config.LATENCY_LEVELS,
        'Total Profit/Loss (€)': [pivot.loc['TOTAL', lat] if lat in pivot.columns else 0 for lat in config.LATENCY_LEVELS],
        'Latency (ms)': [lat / 1000 for lat in config.LATENCY_LEVELS],
        '% of 0 latency': [
            (pivot.loc['TOTAL', lat] / profit_at_zero * 100) if profit_at_zero != 0 and lat in pivot.columns else 0 
            for lat in config.LATENCY_LEVELS
        ]
    })
    
    print(summary_df.to_string(index=False))
    
    # Nota: El conteo de exchange directions requiere recargar datos para cada ISIN
    # Esto se hace mejor en el notebook donde tenemos acceso a DATA_PATH y DATE
    # Por ahora, solo mostramos la tabla pivot y el summary
    
    return pivot, summary_df


def create_decay_chart(money_table_data: List[Dict], save_path: Optional[str] = None) -> None:
    """
    Crea gráficos de decay del profit con latencia.
    
    
    Args:
        money_table_data: Lista de diccionarios con estructura:
                         [{'ISIN': str, 'Latency_us': int, 'Profit_EUR': float}, ...]
        save_path: Ruta opcional para guardar el gráfico
    """
    if not money_table_data:
        print("No data available for Decay Chart.")
        return
    
    money_df = pd.DataFrame(money_table_data)
    
    # Calcular totales por latencia
    totals_by_latency = money_df.groupby('Latency_us')['Profit_EUR'].sum()
    
    latencies_ms = [lat / 1000 for lat in config.LATENCY_LEVELS]
    profits = [totals_by_latency.get(lat, 0) for lat in config.LATENCY_LEVELS]
    
    max_profit = profits[0]
    
    # Check if there are any losses
    has_losses = any(p < 0 for p in profits)
    
    # Calculate percentages
    percentages = [(p / max_profit * 100) if max_profit != 0 else 0 for p in profits]
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Log scale (if all profits are positive)
    if not has_losses and all(p > 0 for p in profits):
        ax1.semilogy(latencies_ms, profits, 'b-o', linewidth=2, markersize=8)
        ax1.set_ylabel('Profit (€, log scale)', fontsize=12)
        ax1.set_title('Profit Decay with Latency (Log Scale)', fontsize=14, fontweight='bold')
    else:
        ax1.plot(latencies_ms, profits, 'b-o', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Break-even')
        ax1.set_ylabel('Profit/Loss (€)', fontsize=12)
        ax1.set_title('Profit Decay with Latency (Linear Scale)', fontsize=14, fontweight='bold')
        ax1.legend()
    
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Percentage of 0 latency profit
    ax2.plot(latencies_ms, percentages, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Break-even')
    ax2.axhline(y=100, color='b', linestyle=':', linewidth=1, alpha=0.5, label='100% (0 latency)')
    ax2.set_xlabel('Latency (ms)', fontsize=12)
    ax2.set_ylabel('% of Profit at 0 Latency', fontsize=12)
    ax2.set_title('Profit Decay as % of 0 Latency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  Gráfico guardado en: {save_path}")
    
    plt.show()
    
    # Imprimir análisis de decay
    print("\nDecay Analysis:")
    print(f"  Maximum profit (0 latency): €{max_profit:,.2f}")
    
    # Find profits at key latencies
    key_latencies = [1000, 10000, 100000]  # 1ms, 10ms, 100ms
    for lat_us in key_latencies:
        if lat_us in totals_by_latency.index:
            profit_at_latency = totals_by_latency[lat_us]
            print(f"  Profit/Loss at {lat_us/1000}ms: €{profit_at_latency:,.2f}")


def identify_top_opportunities(money_table_data: List[Dict], 
                                data_path: str,
                                date: str,
                                n: int = 5) -> pd.DataFrame:
    """
    Identifica las N ISINs más rentables con detalles de oportunidades.
    
    Args:
        money_table_data: Lista de diccionarios con estructura:
                         [{'ISIN': str, 'Latency_us': int, 'Profit_EUR': float}, ...]
        data_path: Ruta al directorio de datos
        date: Fecha en formato YYYY-MM-DD
        n: Número de top ISINs a identificar (default: 5)
        
    Returns:
        DataFrame con información de las top oportunidades
    """
    if not money_table_data:
        print("No data available for Top Opportunities.")
        return pd.DataFrame()
    
    money_df = pd.DataFrame(money_table_data)
    
    # Obtener top N ISINs por profit a latencia 0
    zero_latency = money_df[money_df['Latency_us'] == 0]
    top_isins = zero_latency.nlargest(n, 'Profit_EUR')
    
    print("TOP 5 MOST PROFITABLE ISINs (at 0 latency)")
    print("="*80)
    print()
    
    # Importar funciones necesarias
    from data_loader_module import load_data_for_isin_reference_format
    from consolidator_module import create_consolidated_tape
    from signal_generator_module import detect_arbitrage_opportunities, apply_rising_edge_detection
    
    top_opportunities_list = []
    
    for rank, (_, row) in enumerate(top_isins.iterrows(), 1):
        isin = row['ISIN']
        total_profit = row['Profit_EUR']
        
        # Recargar datos para obtener información detallada
        data_dict = load_data_for_isin_reference_format(data_path, date, isin)
        
        if not data_dict:
            continue
        
        # Verificar que el formato es correcto: Dict[str, Tuple[DataFrame, DataFrame]]
        # create_consolidated_tape espera este formato exacto
        # load_data_for_isin_reference_format ya devuelve este formato
        consolidated = create_consolidated_tape(data_dict)
        
        if consolidated.empty:
            continue
        
        # Detect arbitrage opportunities
        opportunities = detect_arbitrage_opportunities(consolidated)
        
        if opportunities.empty:
            continue
        
        # Apply rising edge detection
        opportunities = apply_rising_edge_detection(opportunities)
        
        if opportunities.empty:
            continue
        
        num_opps = len(opportunities)
        avg_profit = opportunities['total_profit'].mean()
        max_profit = opportunities['total_profit'].max()
        total_qty = opportunities['tradeable_qty'].sum()
        
        # Find best opportunity
        best_opp = opportunities.loc[opportunities['total_profit'].idxmax()]
        
        print(f"{rank}. ISIN: {isin}")
        print(f"   Total Theoretical Profit: €{total_profit:,.2f}")
        print(f"   Number of opportunities: {num_opps}")
        print(f"   Average profit per opportunity: €{avg_profit:,.2f}")
        print(f"   Max profit per opportunity: €{max_profit:,.2f}")
        print(f"   Total tradeable quantity: {total_qty:,.0f} shares")
        print()
        print(f"   Best Opportunity:")
        print(f"     Buy at: {best_opp['buy_exchange']} @ €{best_opp['buy_price']:.4f}")
        print(f"     Sell at: {best_opp['sell_exchange']} @ €{best_opp['sell_price']:.4f}")
        print(f"     Profit per share: €{best_opp['profit_per_share']:.4f}")
        print(f"     Quantity: {best_opp['tradeable_qty']:.0f} shares")
        print(f"     Total profit: €{best_opp['total_profit']:.2f}")
        print()
        
        top_opportunities_list.append({
            'rank': rank,
            'isin': isin,
            'total_profit': total_profit,
            'num_opportunities': num_opps,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'total_qty': total_qty,
            'best_buy_exchange': best_opp['buy_exchange'],
            'best_sell_exchange': best_opp['sell_exchange'],
            'best_buy_price': best_opp['buy_price'],
            'best_sell_price': best_opp['sell_price'],
            'best_profit_per_share': best_opp['profit_per_share'],
            'best_qty': best_opp['tradeable_qty'],
            'best_total_profit': best_opp['total_profit']
        })
    
    # Tabla resumen
    print("="*80)
    print("TOP 5 SUMMARY TABLE")
    print("="*80)
    summary_table = pd.DataFrame({
        'ISIN': top_isins['ISIN'].values,
        'Total Profit at 0 Latency (€)': [f"€{p:,.2f}" for p in top_isins['Profit_EUR'].values]
    })
    print(summary_table.to_string(index=False))
    print()
    
    # Verificaciones de razonabilidad
    print("="*80)
    print("SANITY CHECKS")
    print("="*80)
    print("✓ Checking if profits are reasonable...")
    print()
    
    for _, row in top_isins.iterrows():
        isin = row['ISIN']
        
        # Reload to get average price
        data_dict = load_data_for_isin_reference_format(data_path, date, isin)
        
        if not data_dict:
            continue
        
        # create_consolidated_tape espera Dict[str, Tuple[DataFrame, DataFrame]]
        # que es exactamente lo que devuelve load_data_for_isin_reference_format
        # No necesitamos convertir el formato, ya está correcto
        consolidated = create_consolidated_tape(data_dict)
        
        if consolidated.empty:
            continue
        
        opportunities = detect_arbitrage_opportunities(consolidated)
        opportunities = apply_rising_edge_detection(opportunities)
        
        if opportunities.empty:
            continue
        
        avg_price = (opportunities['buy_price'].mean() + opportunities['sell_price'].mean()) / 2
        avg_profit_per_share = opportunities['profit_per_share'].mean()
        avg_profit_pct = (avg_profit_per_share / avg_price) * 100
        
        print(f"{isin}:")
        print(f"  Average price: €{avg_price:.4f}")
        print(f"  Average profit per share: €{avg_profit_per_share:.4f}")
        print(f"  Average profit %: {avg_profit_pct:.4f}%")
        
        if avg_profit_pct < 1.0:
            print(f"  ✓ Profit percentage looks reasonable (<1%)")
        else:
            print(f"  ⚠ Profit percentage seems high (>1%)")
        print()
    
    return pd.DataFrame(top_opportunities_list)


def generate_summary_answers(money_table_data: List[Dict]) -> Dict:
    """
    Genera respuestas a las 3 preguntas clave del análisis.
    
    Args:
        money_table_data: Lista de diccionarios con estructura:
                         [{'ISIN': str, 'Latency_us': int, 'Profit_EUR': float}, ...]
        
    Returns:
        Dict con las respuestas y métricas clave
    """
    if not money_table_data:
        print("(WARNING) No hay datos disponibles para el resumen.")
        return {}
    
    money_df = pd.DataFrame(money_table_data)
    
    # Calcular totales
    totals_by_latency = money_df.groupby('Latency_us')['Profit_EUR'].sum()
    
    max_profit = totals_by_latency.get(0, 0)
    profit_1ms = totals_by_latency.get(1000, 0)
    profit_10ms = totals_by_latency.get(10000, 0)
    profit_100ms = totals_by_latency.get(100000, 0)
    
    # Count ISINs with opportunities
    isins_with_opps = money_df[money_df['Latency_us'] == 0]
    isins_with_opps = isins_with_opps[isins_with_opps['Profit_EUR'] > 0]
    num_isins = len(isins_with_opps)
    
    # Calculate half-life (50% profit remaining)
    half_profit = max_profit / 2
    half_life_ms = None
    
    for lat_us in config.LATENCY_LEVELS:
        profit = totals_by_latency.get(lat_us, 0)
        if profit <= half_profit:
            half_life_ms = lat_us / 1000
            break
    
    if half_life_ms is None:
        half_life_ms = 100.0  # Default if not reached
    
    # Check for losses
    has_losses = any(p < 0 for p in totals_by_latency.values)
    
    print("="*80)
    print("RESPUESTAS A LAS PREGUNTAS CLAVE")
    print("="*80)
    print()
    
    print("1. ¿Existen aún oportunidades de arbitraje en acciones españolas?")
    if max_profit > 0:
        print(f"   (OK) ¡SÍ! Se encontraron oportunidades de arbitraje con profit teórico total de €{max_profit:,.2f}")
        print(f"   (OK) Número de ISINs con oportunidades: {num_isins}")
    else:
        print("   (WARNING) NO se encontraron oportunidades de arbitraje.")
    print()
    
    print("2. ¿Cuál es el profit teórico máximo (asumiendo latencia 0)?")
    print(f"   Profit teórico máximo: €{max_profit:,.2f}")
    if num_isins > 0:
        top_isin = isins_with_opps.nlargest(1, 'Profit_EUR').iloc[0]
        print(f"   Top ISIN: {top_isin['ISIN']} con €{top_isin['Profit_EUR']:,.2f}")
    print()
    
    print("3. La Curva de 'Decay de Latencia': ¿Qué tan rápido desaparece el profit?")
    print(f"   A 0µs (0ms):     €{max_profit:,.2f} (100.0%)")
    if profit_1ms is not None:
        pct_1ms = (profit_1ms / max_profit * 100) if max_profit != 0 else 0
        print(f"   A 1,000µs (1ms):  €{profit_1ms:,.2f} ({pct_1ms:.1f}%)")
    if profit_10ms is not None:
        pct_10ms = (profit_10ms / max_profit * 100) if max_profit != 0 else 0
        print(f"   A 10,000µs (10ms): €{profit_10ms:,.2f} ({pct_10ms:.1f}%)")
    if profit_100ms is not None:
        pct_100ms = (profit_100ms / max_profit * 100) if max_profit != 0 else 0
        print(f"   A 100,000µs (100ms): €{profit_100ms:,.2f} ({pct_100ms:.1f}%)")
    print()
    print(f"   Vida media (50% del profit restante): ~{half_life_ms:.1f}ms")
    print()
    
    if has_losses:
        print("   (WARNING) Algunas latencias resultaron en pérdidas (profits negativos).")
        print("     Esto indica que las oportunidades de arbitraje pueden convertirse en pérdidas con latencia.")
        print()
    
    print("="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    
    # Retornar métricas en formato dict
    return {
        'max_profit': max_profit,
        'num_isins_with_opps': num_isins,
        'profit_1ms': profit_1ms,
        'profit_10ms': profit_10ms,
        'profit_100ms': profit_100ms,
        'half_life_ms': half_life_ms,
        'has_losses': has_losses,
        'top_isin': isins_with_opps.nlargest(1, 'Profit_EUR').iloc[0]['ISIN'] if num_isins > 0 else None
    }


# ============================================================================
# Clase ArbitrageAnalyzer (compatibilidad con código existente)
# ============================================================================

class ArbitrageAnalyzer:
    """
    Clase wrapper para mantener compatibilidad con código existente.
    
    Las funciones reales están en las funciones del módulo.
    """
    
    def __init__(self):
        """Inicializa el analizador."""
        logger.info("ArbitrageAnalyzer initialized")
    
    def analyze_opportunities(self, signals_df: pd.DataFrame, exec_df: pd.DataFrame = None) -> Dict:
        """
        Análisis básico de oportunidades (compatibilidad).
        
        Args:
            signals_df: DataFrame de señales detectadas
            exec_df: DataFrame de ejecuciones (opcional)
            
        Returns:
            Dict con métricas básicas
        """
        metrics = {}
        
        if signals_df is not None and len(signals_df) > 0:
            total_rising_edges = signals_df.get('is_rising_edge', pd.Series([False])).sum()
            valid_opportunities = signals_df[
                (signals_df.get('is_rising_edge', False)) & 
                (signals_df.get('total_profit', 0) > 0)
            ]
            total_profit = valid_opportunities['total_profit'].sum() if len(valid_opportunities) > 0 else 0
            
            metrics = {
                'total_rising_edges': int(total_rising_edges),
                'total_profit': float(total_profit),
                'num_opportunities': len(valid_opportunities)
            }
        
        return metrics
    
    def estimate_roi(self, metrics: Dict, trading_costs_bps: float = 0.0, capital_eur: float = 100000) -> Dict:
        """
        Estima ROI (compatibilidad).
        
        Args:
            metrics: Dict con métricas
            trading_costs_bps: Costos de trading en basis points
            capital_eur: Capital en EUR
            
        Returns:
            Dict con métricas de ROI
        """
        roi_metrics = {}
        
        if metrics.get('total_profit'):
            gross_profit = metrics['total_profit']
            total_ops = metrics.get('num_opportunities', 0)
            
            if total_ops > 0:
                avg_notional_per_trade = capital_eur / 10
                trading_costs = total_ops * avg_notional_per_trade * (trading_costs_bps / 10000)
                net_profit = gross_profit - trading_costs
                roi_pct = (net_profit / capital_eur * 100) if capital_eur > 0 else 0
                
                roi_metrics = {
                    'gross_profit': gross_profit,
                    'trading_costs': trading_costs,
                    'net_profit': net_profit,
                    'roi_pct': roi_pct
                }
        
        return roi_metrics
    
    def generate_summary_report(self, metrics: Dict, roi_metrics: Dict = None, output_path: str = None) -> str:
        """
        Genera reporte resumen (compatibilidad).
        
        Args:
            metrics: Dict con métricas
            roi_metrics: Dict con métricas de ROI (opcional)
            output_path: Ruta para guardar el reporte (opcional)
            
        Returns:
            String con el reporte
        """
        from datetime import datetime
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DE ANÁLISIS DE ARBITRAJE")
        report_lines.append("=" * 80)
        report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if metrics.get('total_profit'):
            report_lines.append(f"Profit teórico: €{metrics['total_profit']:,.2f}")
        
        if roi_metrics and roi_metrics.get('roi_pct'):
            report_lines.append(f"ROI: {roi_metrics['roi_pct']:.4f}%")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
