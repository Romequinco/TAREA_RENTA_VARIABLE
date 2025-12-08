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
        """Valida señales y calcula estadísticas (no filtra, solo reporta)."""
        rising_edges = signals_df[signals_df.get('is_rising_edge', False)]
        
        stats = {
            'total_signals': len(rising_edges),
            'invalid_spread': (rising_edges['max_bid'] <= rising_edges['min_ask']).sum() if len(rising_edges) > 0 else 0,
            'negative_profit': (rising_edges['theoretical_profit'] < 0).sum() if len(rising_edges) > 0 else 0,
            'zero_qty': (rising_edges['executable_qty'] <= 0).sum() if len(rising_edges) > 0 else 0,
            'same_venue': 0,
            'valid_signals': len(rising_edges),
            'total_profit': rising_edges['total_profit'].sum() if len(rising_edges) > 0 else 0.0,
            'avg_profit': rising_edges['total_profit'].mean() if len(rising_edges) > 0 else 0.0
        }
        
        # Validar mismo venue
        venue_max = rising_edges.get('venue_max_bid', rising_edges.get('max_bid_venue', None))
        venue_min = rising_edges.get('venue_min_ask', rising_edges.get('min_ask_venue', None))
        if venue_max is not None and venue_min is not None:
            stats['same_venue'] = (venue_max == venue_min).sum()
        
        logger.info(f"Validación: {stats['valid_signals']} señales válidas, profit total: €{stats['total_profit']:.2f}")
        return signals_df, stats
    
    @staticmethod
    def filter_edge_cases(signals_df: pd.DataFrame, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """Filtra edge cases (actualmente deshabilitado - mantiene todas las oportunidades)."""
        return signals_df
    
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
        
        # ========================================================================
        # PASO 1: Identificar qué columnas son bids (compradores) y asks (vendedores)
        # ========================================================================
        bid_cols = [col for col in df.columns if col.endswith('_bid') and not col.endswith('_bid_qty')]
        ask_cols = [col for col in df.columns if col.endswith('_ask') and not col.endswith('_ask_qty')]
        
        if not bid_cols or not ask_cols:
            logger.error("No se encontraron columnas de precios (_bid, _ask)")
            return pd.DataFrame()
        
        venues = sorted(set([col.replace('_bid', '').replace('_ask', '') 
                            for col in bid_cols + ask_cols]))
        
        print(f"  Mercados detectados: {venues}")
        print(f"  Total momentos analizados: {len(df):,}")
        
        # ========================================================================
        # PASO 2: En cada momento, encontrar el MEJOR COMPRADOR (MAX bid)
        #         ¿Quién está dispuesto a pagar MÁS por las manzanas?
        # ========================================================================
        print("\n  PASO 2: Buscando el mejor comprador en cada momento...")
        df['max_bid'] = df[bid_cols].max(axis=1, skipna=True)
        df['venue_max_bid'] = df[bid_cols].fillna(-np.inf).idxmax(axis=1).str.replace('_bid', '')
        
        # ========================================================================
        # PASO 3: En cada momento, encontrar el MEJOR VENDEDOR (MIN ask)
        #         ¿Quién está dispuesto a vender por MENOS?
        # ========================================================================
        print("  PASO 3: Buscando el mejor vendedor en cada momento...")
        df['min_ask'] = df[ask_cols].min(axis=1, skipna=True)
        df['venue_min_ask'] = df[ask_cols].fillna(np.inf).idxmin(axis=1).str.replace('_ask', '')
        
        # ========================================================================
        # PASO 4: LA REGLA DE ORO - ¿Hay oportunidad de arbitraje?
        #         MAX(todos los bids) > MIN(todos los asks) → ¡OPORTUNIDAD!
        # ========================================================================
        print("  PASO 4: Aplicando la regla de oro...")
        print("    ¿MAX(bid) > MIN(ask)?")
        
        # Solo comparar si ambos valores existen (no son NaN)
        has_both_values = df['max_bid'].notna() & df['min_ask'].notna()
        
        # La condición de arbitraje: max_bid > min_ask
        arbitrage_condition = (df['max_bid'] > df['min_ask'])
        
        # Señal = 1 si hay oportunidad, 0 si no
        df['signal'] = (has_both_values & arbitrage_condition).astype(int)
        df['is_opportunity'] = df['signal'].astype(bool)
        
        # Reportar resultados
        total_opportunities = df['signal'].sum()
        print(f"\n  ✓ Oportunidades detectadas: {total_opportunities:,} de {len(df):,} momentos")
        print(f"    ({total_opportunities/len(df)*100:.4f}% de los momentos tienen arbitraje)")
        
        # ========================================================================
        # PASO 5: Calcular cantidades y profits (optimizado vectorizado)
        # ========================================================================
        print("\n  PASO 5: Calculando cantidades y ganancias...")
        
        # Extraer cantidades de forma vectorizada usando lookup
        df['bid_qty'] = 0.0
        df['ask_qty'] = 0.0
        
        # Optimización: usar lookup vectorizado en lugar de loop
        for venue in venues:
            bid_qty_col = f'{venue}_bid_qty'
            ask_qty_col = f'{venue}_ask_qty'
            
            if bid_qty_col in df.columns:
                mask = df['venue_max_bid'] == venue
                df.loc[mask, 'bid_qty'] = df.loc[mask, bid_qty_col].fillna(0.0)
            
            if ask_qty_col in df.columns:
                mask = df['venue_min_ask'] == venue
                df.loc[mask, 'ask_qty'] = df.loc[mask, ask_qty_col].fillna(0.0)
        
        # Calcular todo en una sola pasada vectorizada
        df['executable_qty'] = np.minimum(df['bid_qty'].fillna(0.0), df['ask_qty'].fillna(0.0))
        df['theoretical_profit'] = df['max_bid'] - df['min_ask']
        df['total_profit'] = np.where(df['signal'] == 1, 
                                     df['theoretical_profit'] * df['executable_qty'], 
                                     0.0)
        df['remaining_bid_qty'] = df['bid_qty'].fillna(0.0)
        df['remaining_ask_qty'] = df['ask_qty'].fillna(0.0)
        
        # Asegurar profit=0 cuando no hay señal
        df.loc[df['signal'] == 0, ['theoretical_profit', 'executable_qty']] = 0.0
        
        # PASO 6: Aplicar Rising Edge Detection y validar
        print("  Aplicando Rising Edge Detection...")
        df = self.apply_rising_edge(df)
        
        print("  Validando señales...")
        df, validation_stats = self.validate_signals(df)
        
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
        signals_df = self.generate_signals(consolidated_tape)
        
        if len(signals_df) == 0 or not executed_trades:
            return signals_df
        
        # Optimización: usar merge_asof en lugar de iterrows (mucho más rápido)
        exec_df = pd.DataFrame(executed_trades)
        if len(exec_df) == 0:
            return signals_df
        
        signals_df['_consumed_execution_id'] = None
        
        # Filtrar por ISIN si se proporciona
        if isin and 'isin' in exec_df.columns:
            exec_df = exec_df[exec_df['isin'] == isin]
        
        if len(exec_df) == 0:
            return signals_df
        
        # Merge_asof para encontrar ejecuciones anteriores a cada señal (optimizado)
        opps = signals_df[signals_df['is_opportunity']].copy()
        if len(opps) == 0:
            return signals_df
        
        # Crear clave de venue pair para matching
        opps['venue_pair'] = (opps.get('venue_max_bid', '') + '_' + 
                             opps.get('venue_min_ask', ''))
        exec_df['venue_pair'] = (exec_df.get('venue_max_bid', '') + '_' + 
                                 exec_df.get('venue_min_ask', ''))
        
        # Merge_asof para encontrar última ejecución antes de cada oportunidad
        merged = pd.merge_asof(
            opps.sort_values('epoch'),
            exec_df.sort_values('epoch'),
            on='epoch',
            by='venue_pair',
            direction='backward',
            suffixes=('', '_exec')
        )
        
        # Actualizar cantidades remanentes donde hay ejecuciones previas
        has_exec = merged['executed_qty_exec'].notna()
        if has_exec.any():
            exec_mask = signals_df.index.isin(merged[has_exec].index)
            executed_qty = merged.loc[has_exec, 'executed_qty_exec'].values
            
            signals_df.loc[exec_mask, 'remaining_bid_qty'] = np.maximum(0, 
                signals_df.loc[exec_mask, 'bid_qty'].values - executed_qty)
            signals_df.loc[exec_mask, 'remaining_ask_qty'] = np.maximum(0,
                signals_df.loc[exec_mask, 'ask_qty'].values - executed_qty)
            
            signals_df.loc[exec_mask, 'executable_qty'] = np.minimum(
                signals_df.loc[exec_mask, 'remaining_bid_qty'].values,
                signals_df.loc[exec_mask, 'remaining_ask_qty'].values
            )
            signals_df.loc[exec_mask, 'tradeable_qty'] = signals_df.loc[exec_mask, 'executable_qty']
            signals_df.loc[exec_mask, 'total_profit'] = (
                signals_df.loc[exec_mask, 'theoretical_profit'].values * 
                signals_df.loc[exec_mask, 'executable_qty'].values
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
