"""
================================================================================
signal_generator.py - Módulo de Detección de Señales de Arbitraje
================================================================================

Responsabilidades:
- Calcular Global Max Bid y Global Min Ask en cada timestamp
- Detectar condición de arbitraje: Bid_max > Ask_min
- Calcular profit teórico por oportunidad
- Implementar Rising Edge Detection (evitar doble conteo)
- Identificar venues involucrados en cada oportunidad
- NUEVO: Gráficos corregidos y funcionales

Condición de Arbitraje:
    Global_Max_Bid > Global_Min_Ask
    
Profit Teórico:
    (Max_Bid - Min_Ask) * Min(Bid_Qty, Ask_Qty)

Rising Edge:
    Si una oportunidad persiste por N snapshots, solo contarla la primera vez
    que aparece (rising edge). Si desaparece y reaparece, es nueva oportunidad.

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from config_module import config

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Detecta oportunidades de arbitraje en el consolidated tape.
    
    El sistema busca instantes donde se puede comprar en un venue y
    vender en otro simultáneamente con profit positivo.
    
    REQUISITO 1: Registra y excluye oportunidades ya ejecutadas.
    REQUISITO 2: Soporta ejecución parcial con cantidades (bid/ask size).
    """
    
    def __init__(self):
        """
        Inicializa el generador de señales con tracking de ejecuciones.
        """
        self.executed_trades = []  # Lista de trades ejecutados
        self.execution_id_counter = 0  # Contador para IDs de ejecución
    
    def detect_opportunities(self, 
                            consolidated_tape: pd.DataFrame, 
                            executed_trades: list = None,
                            isin: str = None) -> pd.DataFrame:
        """
        Detecta todas las oportunidades de arbitraje en el tape.
        
        REQUISITO 1: Excluye oportunidades ya ejecutadas de snapshots posteriores.
        REQUISITO 2: Calcula size_executable = min(bid_size, ask_size) y soporta ejecución parcial.
        
        Proceso:
        1. Para cada timestamp, calcular Global Max Bid y Global Min Ask
        2. Identificar venues del max bid y min ask
        3. Detectar si Max Bid > Min Ask (arbitrage condition)
        4. Calcular profit teórico y cantidad ejecutable
        5. Aplicar Rising Edge Detection
        6. Filtrar oportunidades ya ejecutadas (REQUISITO 1)
        
        Args:
            consolidated_tape: DataFrame con columnas [epoch, XMAD_bid, XMAD_ask, ...]
            executed_trades: Lista de dicts con trades ejecutados (opcional)
                           Cada dict debe tener: epoch, venue_max_bid, venue_min_ask, 
                           executed_qty, execution_id
            isin: ISIN del instrumento (opcional, para tracking)
            
        Returns:
            DataFrame con columnas:
            - epoch: Timestamp
            - global_max_bid: Mejor bid de todos los venues
            - global_min_ask: Mejor ask de todos los venues
            - venue_max_bid: Venue con el mejor bid
            - venue_min_ask: Venue con el mejor ask
            - bid_qty: Cantidad disponible en el mejor bid
            - ask_qty: Cantidad disponible en el mejor ask
            - executable_qty: REQUISITO 2: min(bid_size, ask_size)
            - remaining_bid_qty: REQUISITO 2: Cantidad remanente en bid después de ejecución
            - remaining_ask_qty: REQUISITO 2: Cantidad remanente en ask después de ejecución
            - is_opportunity: Boolean (True si Bid > Ask)
            - theoretical_profit: Profit por unidad
            - total_profit: Profit total (price * quantity)
            - is_rising_edge: Boolean (True si es primera aparición)
            - _consumed_execution_id: REQUISITO 1: ID de ejecución si fue consumida (None si nueva)
        """
        print("\n" + "=" * 80)
        print("DETECCIÓN DE SEÑALES DE ARBITRAJE")
        print("=" * 80)
        
        df = consolidated_tape.copy()
        
        # ====================================================================
        # PASO 1: Extraer columnas de bids y asks
        # ====================================================================
        bid_cols = [col for col in df.columns if col.endswith('_bid')]
        ask_cols = [col for col in df.columns if col.endswith('_ask')]
        
        venues = [col.replace('_bid', '') for col in bid_cols]
        
        print(f"  Venues detectados: {venues}")
        print(f"  Total snapshots a analizar: {len(df):,}")
        
        # ====================================================================
        # PASO 2: Calcular Global Max Bid y Global Min Ask por fila
        # ====================================================================
        print("\n  Calculando Global Max Bid y Global Min Ask...")
        
        # Extraer solo las columnas de bids y asks
        bids_df = df[bid_cols]
        asks_df = df[ask_cols]
        
        # Global Max Bid: Máximo de todos los bids en cada timestamp
        df['global_max_bid'] = bids_df.max(axis=1)
        
        # Global Min Ask: Mínimo de todos los asks en cada timestamp
        df['global_min_ask'] = asks_df.min(axis=1)
        
        # ====================================================================
        # PASO 3: Identificar venues del max bid y min ask
        # ====================================================================
        print("  Identificando venues de cada oportunidad...")
        
        # Para cada fila, encontrar qué venue tiene el max bid
        df['venue_max_bid'] = bids_df.idxmax(axis=1).str.replace('_bid', '')
        
        # Para cada fila, encontrar qué venue tiene el min ask
        df['venue_min_ask'] = asks_df.idxmin(axis=1).str.replace('_ask', '')
        
        # ====================================================================
        # PASO 4: Extraer cantidades disponibles
        # ====================================================================
        print("  Extrayendo cantidades disponibles...")
        
        # Función para obtener cantidad del venue correspondiente
        def get_bid_qty(row):
            venue = row['venue_max_bid']
            qty_col = f'{venue}_bid_qty'
            return row[qty_col] if qty_col in df.columns else 0
        
        def get_ask_qty(row):
            venue = row['venue_min_ask']
            qty_col = f'{venue}_ask_qty'
            return row[qty_col] if qty_col in df.columns else 0
        
        df['bid_qty'] = df.apply(get_bid_qty, axis=1)
        df['ask_qty'] = df.apply(get_ask_qty, axis=1)
        
        # ====================================================================
        # PASO 5: Detectar condición de arbitraje
        # ====================================================================
        print("  Detectando condición de arbitraje (Bid > Ask)...")
        
        # Condición: Global Max Bid > Global Min Ask
        df['is_opportunity'] = df['global_max_bid'] > df['global_min_ask']
        
        # ====================================================================
        # PASO 6: Calcular profit teórico y cantidades ejecutables
        # ====================================================================
        print("  Calculando profit teórico...")
        
        # Profit por unidad: (Max Bid - Min Ask)
        df['theoretical_profit'] = df['global_max_bid'] - df['global_min_ask']
        
        # REQUISITO 2: size_executable = min(bid_size, ask_size)
        df['executable_qty'] = np.minimum(df['bid_qty'], df['ask_qty'])
        
        # REQUISITO 2: Inicializar cantidades remanentes (antes de aplicar ejecuciones)
        df['remaining_bid_qty'] = df['bid_qty'].copy()
        df['remaining_ask_qty'] = df['ask_qty'].copy()
        
        # Profit total: Profit por unidad * Cantidad ejecutable
        df['total_profit'] = df['theoretical_profit'] * df['executable_qty']
        
        # Si no hay oportunidad, profit = 0
        df.loc[~df['is_opportunity'], 'theoretical_profit'] = 0
        df.loc[~df['is_opportunity'], 'total_profit'] = 0
        df.loc[~df['is_opportunity'], 'executable_qty'] = 0
        
        # ====================================================================
        # PASO 7: Rising Edge Detection
        # ====================================================================
        print("  Aplicando Rising Edge Detection...")
        
        # Rising edge: Primera aparición de una oportunidad
        # Si is_opportunity cambia de False a True, es rising edge
        
        df['is_rising_edge'] = False
        
        # Comparar con el snapshot anterior
        df['prev_opportunity'] = df['is_opportunity'].shift(1).fillna(False)
        
        # Rising edge = oportunidad actual AND no había oportunidad anterior
        df['is_rising_edge'] = df['is_opportunity'] & ~df['prev_opportunity']
        
        # Limpiar columna temporal
        df = df.drop('prev_opportunity', axis=1)
        
        # ====================================================================
        # PASO 8: REQUISITO 1 - Filtrar oportunidades ya ejecutadas
        # ====================================================================
        print("  Filtrando oportunidades ya ejecutadas...")
        
        # Inicializar columna _consumed_execution_id
        df['_consumed_execution_id'] = None
        
        if executed_trades and len(executed_trades) > 0:
            # Crear DataFrame de trades ejecutados para merge eficiente
            exec_df = pd.DataFrame(executed_trades)
            
            # Para cada oportunidad, verificar si ya fue ejecutada
            # Una oportunidad está ejecutada si:
            # - Mismo epoch (o posterior)
            # - Mismos venues (venue_max_bid y venue_min_ask)
            # - Mismo ISIN (si está disponible)
            
            for idx, row in df[df['is_opportunity']].iterrows():
                # Buscar trades ejecutados que coincidan
                matching_execs = exec_df[
                    (exec_df['epoch'] <= row['epoch']) &
                    (exec_df['venue_max_bid'] == row['venue_max_bid']) &
                    (exec_df['venue_min_ask'] == row['venue_min_ask'])
                ]
                
                if isin and 'isin' in exec_df.columns:
                    matching_execs = matching_execs[matching_execs['isin'] == isin]
                
                if len(matching_execs) > 0:
                    # Oportunidad ya fue ejecutada
                    # Usar el execution_id más reciente
                    latest_exec = matching_execs.loc[matching_execs['epoch'].idxmax()]
                    df.at[idx, '_consumed_execution_id'] = latest_exec.get('execution_id', None)
                    
                    # REQUISITO 2: Ajustar cantidades remanentes por ejecución parcial
                    executed_qty = latest_exec.get('executed_qty', 0)
                    df.at[idx, 'remaining_bid_qty'] = max(0, row['bid_qty'] - executed_qty)
                    df.at[idx, 'remaining_ask_qty'] = max(0, row['ask_qty'] - executed_qty)
                    df.at[idx, 'executable_qty'] = np.minimum(
                        df.at[idx, 'remaining_bid_qty'],
                        df.at[idx, 'remaining_ask_qty']
                    )
                    
                    # Recalcular profit con cantidades ajustadas
                    df.at[idx, 'total_profit'] = df.at[idx, 'theoretical_profit'] * df.at[idx, 'executable_qty']
            
            # Filtrar oportunidades que ya fueron completamente ejecutadas
            # (si executable_qty <= 0 después de ajustar)
            fully_executed = (df['_consumed_execution_id'].notna()) & (df['executable_qty'] <= 0)
            if fully_executed.any():
                logger.info(f"    Excluidas {fully_executed.sum():,} oportunidades ya completamente ejecutadas")
        
        # ====================================================================
        # PASO 9: Filtrar oportunidades válidas (nuevas o parcialmente ejecutables)
        # ====================================================================
        print("  Filtrando oportunidades válidas...")
        
        # Filtrar por threshold mínimo de profit Y que tengan cantidad ejecutable > 0
        valid_opportunities = df[
            (df['is_rising_edge']) & 
            (df['total_profit'] >= config.MIN_THEORETICAL_PROFIT) &
            (df['executable_qty'] > 0)  # REQUISITO 2: Solo oportunidades con cantidad ejecutable
        ].copy()
        
        # ====================================================================
        # RESUMEN
        # ====================================================================
        total_snapshots = len(df)
        total_opportunities = df['is_opportunity'].sum()
        rising_edges = df['is_rising_edge'].sum()
        valid_rising_edges = len(valid_opportunities)
        
        print(f"\n  RESULTADOS:")
        print(f"    - Total snapshots analizados: {total_snapshots:,}")
        print(f"    - Snapshots con arbitraje: {total_opportunities:,} ({total_opportunities/total_snapshots*100:.2f}%)")
        print(f"    - Rising edges detectados: {rising_edges:,}")
        print(f"    - Rising edges válidos (>€{config.MIN_THEORETICAL_PROFIT}): {valid_rising_edges:,}")
        
        if valid_rising_edges > 0:
            total_profit = valid_opportunities['total_profit'].sum()
            avg_profit = valid_opportunities['total_profit'].mean()
            max_profit = valid_opportunities['total_profit'].max()
            
            print(f"    - Profit teórico total: €{total_profit:.2f}")
            print(f"    - Profit medio por oportunidad: €{avg_profit:.2f}")
            print(f"    - Profit máximo: €{max_profit:.2f}")
        
        # Seleccionar columnas relevantes (incluyendo nuevas columnas de REQUISITO 1 y 2)
        cols_to_keep = [
            'epoch', 'global_max_bid', 'global_min_ask',
            'venue_max_bid', 'venue_min_ask',
            'bid_qty', 'ask_qty', 'executable_qty',
            'remaining_bid_qty', 'remaining_ask_qty',  # REQUISITO 2
            'is_opportunity', 'theoretical_profit', 'total_profit',
            'is_rising_edge', '_consumed_execution_id'  # REQUISITO 1
        ]
        
        # Solo incluir columnas que existen
        available_cols = [col for col in cols_to_keep if col in df.columns]
        signals_df = df[available_cols].copy()
        
        return signals_df
    
    @staticmethod
    def analyze_venue_pairs(signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza qué pares de venues generan más oportunidades.
        
        Args:
            signals_df: DataFrame con señales detectadas
            
        Returns:
            DataFrame con estadísticas por par de venues
        """
        print("\n" + "=" * 80)
        print("ANÁLISIS DE PARES DE VENUES")
        print("=" * 80)
        
        # Filtrar solo rising edges válidos
        opportunities = signals_df[
            (signals_df['is_rising_edge']) & 
            (signals_df['total_profit'] > 0)
        ].copy()
        
        if len(opportunities) == 0:
            print("  No hay oportunidades válidas para analizar")
            return pd.DataFrame()
        
        # Crear columna con par de venues
        opportunities['venue_pair'] = (
            'Buy@' + opportunities['venue_min_ask'] + 
            ' / Sell@' + opportunities['venue_max_bid']
        )
        
        # Agrupar por par de venues
        pairs_stats = opportunities.groupby('venue_pair').agg({
            'total_profit': ['count', 'sum', 'mean', 'max'],
            'theoretical_profit': 'mean',
            'executable_qty': 'mean'
        }).reset_index()
        
        # Renombrar columnas
        pairs_stats.columns = [
            'Venue Pair', 
            'Count', 
            'Total Profit (€)', 
            'Avg Profit (€)', 
            'Max Profit (€)',
            'Avg Price Diff (€)',
            'Avg Qty'
        ]
        
        # Ordenar por profit total
        pairs_stats = pairs_stats.sort_values('Total Profit (€)', ascending=False)
        
        print("\nTop 10 Pares de Venues:")
        print(pairs_stats.head(10).to_string(index=False))
        
        return pairs_stats
    
    @staticmethod
    def visualize_signals(signals_df: pd.DataFrame, isin: str, max_points: int = 10000):
        """
        Visualiza las señales de arbitraje detectadas.
        
        NUEVO: Gráficos corregidos y funcionales
        
        Args:
            signals_df: DataFrame con señales
            isin: ISIN para el título
            max_points: Máximo de puntos a plotear
        """
        print("\nGenerando visualizaciones de señales...")
        
        # Sampling
        if len(signals_df) > max_points:
            # Tomar muestra aleatoria pero mantener índices originales
            sample_indices = np.sort(np.random.choice(signals_df.index, size=max_points, replace=False))
            sample_df = signals_df.loc[sample_indices].copy()
        else:
            sample_df = signals_df.copy()
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Crear índice numérico para plotting
        x_values = range(len(sample_df))
        
        # ====================================================================
        # Plot 1: Global Max Bid vs Global Min Ask
        # ====================================================================
        ax1 = axes[0]
        
        # Plotear precios
        ax1.plot(x_values, sample_df['global_max_bid'].values, 
                label='Global Max Bid', color='green', alpha=0.7, linewidth=1.5)
        ax1.plot(x_values, sample_df['global_min_ask'].values, 
                label='Global Min Ask', color='red', alpha=0.7, linewidth=1.5)
        
        # Marcar oportunidades
        opportunities_mask = sample_df['is_opportunity']
        if opportunities_mask.any():
            opp_indices = [i for i, v in enumerate(opportunities_mask) if v]
            opp_bids = sample_df.loc[opportunities_mask, 'global_max_bid'].values
            
            ax1.scatter(opp_indices, opp_bids,
                       color='gold', s=30, alpha=0.6, label='Arbitrage Opportunity',
                       zorder=5)
        
        ax1.set_title(f'Global Best Prices - ISIN: {isin}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Snapshot Index')
        ax1.set_ylabel('Price (€)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ====================================================================
        # Plot 2: Theoretical Profit Over Time
        # ====================================================================
        ax2 = axes[1]
        
        # Solo plotear cuando hay oportunidad
        profit_mask = sample_df['is_opportunity'] & (sample_df['theoretical_profit'] > 0)
        
        if profit_mask.any():
            profit_indices = [i for i, v in enumerate(profit_mask) if v]
            profit_values = sample_df.loc[profit_mask, 'theoretical_profit'].values * 10000  # En basis points
            
            ax2.scatter(profit_indices, profit_values,
                       color='blue', alpha=0.6, s=30)
            ax2.set_title('Theoretical Profit per Unit (basis points)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Snapshot Index')
            ax2.set_ylabel('Profit (bps)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No opportunities detected', 
                    ha='center', va='center', fontsize=14,
                    transform=ax2.transAxes)
            ax2.set_xlabel('Snapshot Index')
            ax2.set_ylabel('Profit (bps)')
        
        # ====================================================================
        # Plot 3: Cumulative Profit (Rising Edges only)
        # ====================================================================
        ax3 = axes[2]
        
        rising_edges = signals_df[signals_df['is_rising_edge']].copy()
        
        if len(rising_edges) > 0:
            # Ordenar por índice para cumsum correcto
            rising_edges = rising_edges.sort_index()
            cumulative_profit = rising_edges['total_profit'].cumsum().values
            
            x_cum = range(len(rising_edges))
            
            ax3.plot(x_cum, cumulative_profit,
                    color='darkgreen', linewidth=2)
            ax3.fill_between(x_cum, cumulative_profit,
                            alpha=0.3, color='green')
            ax3.set_title('Cumulative Theoretical Profit (€)', 
                         fontsize=14, fontweight='bold')
            ax3.set_xlabel('Opportunity Number')
            ax3.set_ylabel('Cumulative Profit (€)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No rising edges detected', 
                    ha='center', va='center', fontsize=14,
                    transform=ax3.transAxes)
            ax3.set_xlabel('Opportunity Number')
            ax3.set_ylabel('Cumulative Profit (€)')
        
        plt.tight_layout()
        
        # Guardar y mostrar
        output_path = config.FIGURES_DIR / f'signals_{isin}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Visualización guardada en: {output_path}")
        
        try:
            plt.show(block=False)
            plt.pause(0.5)  # Pausa más larga para asegurar renderizado
            print(f"  [OK] Gráficas mostradas en ventana")
        except Exception as e:
            logger.warning(f"  No se pudo mostrar gráficas interactivas: {e}")
            print(f"  [INFO] Gráficas guardadas en: {output_path}")
        
        # No cerrar inmediatamente para que el usuario pueda verlas
        # plt.close()  # Comentado para permitir visualización
        
        print("  Visualizaciones generadas")
    
    @staticmethod
    def export_opportunities(signals_df: pd.DataFrame, output_path: str = None):
        """
        Exporta las oportunidades detectadas a CSV.
        
        REQUISITO 2: Añade tabla opportunities.csv mostrando qué se ejecutó.
        
        Args:
            signals_df: DataFrame con señales
            output_path: Path para guardar el CSV (opcional)
        """
        # Filtrar solo rising edges con cantidad ejecutable > 0
        opportunities = signals_df[
            (signals_df['is_rising_edge']) & 
            (signals_df['total_profit'] > 0) &
            (signals_df.get('executable_qty', pd.Series([1]*len(signals_df))) > 0)
        ].copy()
        
        if len(opportunities) == 0:
            print("  No hay oportunidades para exportar")
            return
        
        if output_path is None:
            output_path = config.OUTPUT_DIR / "opportunities.csv"
        
        # Asegurar que el directorio existe
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # REQUISITO 2: Añadir columnas de ejecución si no existen
        if 'executed_qty' not in opportunities.columns:
            opportunities['executed_qty'] = 0
        if 'remaining_bid_qty' not in opportunities.columns:
            opportunities['remaining_bid_qty'] = opportunities.get('bid_qty', 0)
        if 'remaining_ask_qty' not in opportunities.columns:
            opportunities['remaining_ask_qty'] = opportunities.get('ask_qty', 0)
        
        # Exportar
        opportunities.to_csv(output_path, index=False)
        
        print(f"\n  Oportunidades exportadas a: {output_path}")
        print(f"    Total oportunidades: {len(opportunities):,}")
        
        # REQUISITO 2: Mostrar resumen de ejecuciones
        if '_consumed_execution_id' in opportunities.columns:
            executed_count = opportunities['_consumed_execution_id'].notna().sum()
            if executed_count > 0:
                print(f"    Oportunidades ya ejecutadas: {executed_count:,}")
                print(f"    Oportunidades nuevas: {len(opportunities) - executed_count:,}")