"""
================================================================================
latency_simulator.py - Simulación Realista de Latencia (Time Machine)
================================================================================

OBJETIVO:
Simular el efecto de la latencia en las oportunidades de arbitraje detectadas.

CONCEPTO CLAVE - TIME MACHINE:
Cuando detectamos una oportunidad en el timestamp T, la orden NO se ejecuta
instantáneamente. Hay que viajar en el tiempo hacia ADELANTE por la latencia:
    T_execution = T_detection + latency

En T_execution, los precios pueden haber cambiado, reduciendo o eliminando
el profit.

LATENCIAS TÍPICAS (microsegundos):
- Co-location (mismo datacenter): 10-50 us
- Cross-venue (diferentes datacenters): 100-500 us
- HFT agresivo: 1-10 us
- Retail: >1000 us

PROCESO:
1. Para cada oportunidad detectada en timestamp T
2. Buscar los precios en T + latency (time-shifted lookup)
3. Re-calcular profit con los nuevos precios
4. Clasificar: Profitable / Break-even / Loss

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


class LatencySimulator:
    """
    Simula el efecto de latencia en oportunidades de arbitraje.
    
    Implementa el "Time Machine": viaja hacia adelante en el tiempo
    para obtener los precios reales en el momento de ejecución.
    """
    
    def __init__(self, latency_us: int = 100):
        """
        Args:
            latency_us: Latencia en microsegundos (default: 100 us)
        """
        self.latency_us = latency_us
        self.latency_ns = latency_us * 1000  # Convertir a nanosegundos
        
        logger.info(f"LatencySimulator initialized: latency={latency_us} us")
    
    def simulate_execution(self, 
                          signals_df: pd.DataFrame,
                          consolidated_tape: pd.DataFrame,
                          isin: str = None) -> Tuple[pd.DataFrame, list]:
        """
        Simula la ejecución de oportunidades considerando latencia.
        
        REQUISITO 1: Registra trades ejecutados en executed_trades.
        REQUISITO 2: Soporta ejecución parcial con cantidades.
        
        ALGORITMO:
        1. Para cada oportunidad (rising edge) en timestamp T
        2. Calcular T_execution = T + latency
        3. Buscar los precios en el consolidated tape en T_execution
        4. Re-calcular profit con precios time-shifted
        5. Comparar profit original vs profit real
        6. Registrar trade ejecutado si es profitable
        
        Args:
            signals_df: DataFrame de signal_generator.detect_opportunities()
                       Debe contener: [epoch, global_max_bid, global_min_ask,
                                      venue_max_bid, venue_min_ask, 
                                      theoretical_profit, is_rising_edge,
                                      executable_qty, remaining_bid_qty, remaining_ask_qty]
            
            consolidated_tape: DataFrame con todos los precios por venue
                              [epoch, XMAD_bid, XMAD_ask, ...]
            isin: ISIN del instrumento (opcional, para tracking)
        
        Returns:
            Tuple de (DataFrame, list):
            - DataFrame con columnas adicionales:
              - execution_epoch: T + latency
              - execution_bid: Precio bid en T_execution en venue_max_bid
              - execution_ask: Precio ask en T_execution en venue_min_ask
              - executed_qty: REQUISITO 2: Cantidad realmente ejecutada
              - real_profit: Profit después de latencia
              - profit_loss: Pérdida debido a latencia (original - real)
              - is_profitable: Boolean (True si real_profit > 0)
              - profit_category: 'Profitable' / 'Break-even' / 'Loss'
              - execution_id: REQUISITO 1: ID único de ejecución
            - list: Lista de dicts con trades ejecutados (REQUISITO 1)
        """
        
        print("\n" + "=" * 80)
        print("SIMULACIÓN DE LATENCIA (TIME MACHINE)")
        print("=" * 80)
        print(f"  Latencia configurada: {self.latency_us} us")
        
        # ====================================================================
        # PASO 1: Filtrar solo rising edges (oportunidades únicas)
        # ====================================================================
        opportunities = signals_df[signals_df['is_rising_edge']].copy()
        
        if len(opportunities) == 0:
            logger.warning("  No hay rising edges para simular")
            return pd.DataFrame(), []  # Retornar tupla vacía
        
        print(f"  Total oportunidades a simular: {len(opportunities):,}")
        
        # ====================================================================
        # PASO 2: Calcular timestamp de ejecución (T + latency)
        # ====================================================================
        opportunities['execution_epoch'] = opportunities['epoch'] + self.latency_ns
        
        # ====================================================================
        # PASO 3: Preparar consolidated tape para búsqueda eficiente
        # ====================================================================
        # Ordenar por epoch para merge_asof
        tape_sorted = consolidated_tape.sort_values('epoch').reset_index(drop=True)
        
        print(f"  Consolidated tape: {len(tape_sorted):,} timestamps")
        print(f"  Rango temporal: {tape_sorted['epoch'].min()} to {tape_sorted['epoch'].max()}")
        
        # ====================================================================
        # PASO 4: Time Machine - Buscar precios en T_execution
        # ====================================================================
        print("\n  [TIME MACHINE] Buscando precios en T + latency...")
        
        # Merge asof: Para cada execution_epoch, encontrar el snapshot 
        # más cercano ANTERIOR en el tape (direction='backward')
        opportunities_with_execution = pd.merge_asof(
            opportunities.sort_values('execution_epoch'),
            tape_sorted,
            left_on='execution_epoch',
            right_on='epoch',
            direction='backward',
            suffixes=('', '_exec')
        )
        
        print(f"  [OK] Precios encontrados para {len(opportunities_with_execution):,} oportunidades")
        
        # ====================================================================
        # PASO 5: Extraer precios de ejecución de los venues correctos
        # ====================================================================
        print("  Extrayendo precios de ejecución por venue...")
        
        def get_execution_bid(row):
            """
            Obtiene el precio bid del venue donde detectamos el max bid,
            pero en el timestamp de ejecución (T + latency)
            """
            venue = row['venue_max_bid']
            bid_col = f'{venue}_bid'
            return row[bid_col] if bid_col in opportunities_with_execution.columns else np.nan
        
        def get_execution_ask(row):
            """
            Obtiene el precio ask del venue donde detectamos el min ask,
            pero en el timestamp de ejecución (T + latency)
            """
            venue = row['venue_min_ask']
            ask_col = f'{venue}_ask'
            return row[ask_col] if ask_col in opportunities_with_execution.columns else np.nan
        
        opportunities_with_execution['execution_bid'] = opportunities_with_execution.apply(
            get_execution_bid, axis=1
        )
        
        opportunities_with_execution['execution_ask'] = opportunities_with_execution.apply(
            get_execution_ask, axis=1
        )
        
        # ====================================================================
        # PASO 6: Calcular profit real después de latencia
        # ====================================================================
        print("  Calculando profit real después de latencia...")
        
        # Profit real = execution_bid - execution_ask
        # (Vendemos al bid, compramos al ask)
        opportunities_with_execution['real_profit'] = (
            opportunities_with_execution['execution_bid'] - 
            opportunities_with_execution['execution_ask']
        )
        
        # REQUISITO 2: Cantidad ejecutable (usar executable_qty o remaining si existe)
        if 'executable_qty' in opportunities_with_execution.columns:
            exec_qty_col = 'executable_qty'
        elif 'remaining_bid_qty' in opportunities_with_execution.columns and 'remaining_ask_qty' in opportunities_with_execution.columns:
            # Calcular cantidad ejecutable desde remanentes
            opportunities_with_execution['executable_qty'] = np.minimum(
                opportunities_with_execution['remaining_bid_qty'],
                opportunities_with_execution['remaining_ask_qty']
            )
            exec_qty_col = 'executable_qty'
        else:
            # Fallback: usar bid_qty o ask_qty mínimo
            opportunities_with_execution['executable_qty'] = np.minimum(
                opportunities_with_execution.get('bid_qty', 0),
                opportunities_with_execution.get('ask_qty', 0)
            )
            exec_qty_col = 'executable_qty'
        
        # REQUISITO 2: Cantidad realmente ejecutada (puede ser parcial)
        # Por ahora, ejecutamos toda la cantidad ejecutable si es profitable
        opportunities_with_execution['executed_qty'] = np.where(
            opportunities_with_execution['real_profit'] > 0.0001,  # Solo si es profitable
            opportunities_with_execution[exec_qty_col],
            0
        )
        
        # Profit total real: Profit por unidad * Cantidad ejecutada
        opportunities_with_execution['real_total_profit'] = (
            opportunities_with_execution['real_profit'] * 
            opportunities_with_execution['executed_qty']
        )
        
        # Pérdida debido a latencia
        opportunities_with_execution['profit_loss'] = (
            opportunities_with_execution['theoretical_profit'] - 
            opportunities_with_execution['real_profit']
        )
        
        opportunities_with_execution['profit_loss_total'] = (
            opportunities_with_execution['total_profit'] - 
            opportunities_with_execution['real_total_profit']
        )
        
        # ====================================================================
        # PASO 7: Clasificar oportunidades
        # ====================================================================
        print("  Clasificando oportunidades...")
        
        # Oportunidad es profitable si real_profit > threshold mínimo
        min_threshold = 0.0001  # 0.01 céntimos (1 basis point)
        
        opportunities_with_execution['is_profitable'] = (
            opportunities_with_execution['real_profit'] > min_threshold
        )
        
        # Categorías
        def categorize_profit(row):
            if pd.isna(row['real_profit']):
                return 'Unknown'
            elif row['real_profit'] > min_threshold:
                return 'Profitable'
            elif row['real_profit'] >= -min_threshold:
                return 'Break-even'
            else:
                return 'Loss'
        
        opportunities_with_execution['profit_category'] = opportunities_with_execution.apply(
            categorize_profit, axis=1
        )
        
        # ====================================================================
        # REQUISITO 1: Registrar trades ejecutados
        # ====================================================================
        print("  Registrando trades ejecutados...")
        
        executed_trades = []
        execution_id_counter = 0
        
        # Filtrar solo oportunidades profitable que fueron ejecutadas
        profitable_executions = opportunities_with_execution[
            (opportunities_with_execution['profit_category'] == 'Profitable') &
            (opportunities_with_execution['executed_qty'] > 0)
        ].copy()
        
        for idx, row in profitable_executions.iterrows():
            execution_id_counter += 1
            execution_id = f"EXEC_{execution_id_counter:08d}"
            
            # REQUISITO 1: Registrar trade ejecutado
            executed_trade = {
                'execution_id': execution_id,
                'epoch': row['epoch'],
                'execution_epoch': row.get('execution_epoch', row['epoch']),
                'venue_max_bid': row['venue_max_bid'],
                'venue_min_ask': row['venue_min_ask'],
                'executed_qty': row['executed_qty'],
                'execution_bid': row.get('execution_bid', row.get('global_max_bid')),
                'execution_ask': row.get('execution_ask', row.get('global_min_ask')),
                'real_profit': row['real_profit'],
                'real_total_profit': row['real_total_profit']
            }
            
            if isin:
                executed_trade['isin'] = isin
            
            executed_trades.append(executed_trade)
            
            # Añadir execution_id al DataFrame
            opportunities_with_execution.at[idx, 'execution_id'] = execution_id
        
        # Inicializar execution_id como None para oportunidades no ejecutadas
        if 'execution_id' not in opportunities_with_execution.columns:
            opportunities_with_execution['execution_id'] = None
        
        print(f"    Trades ejecutados registrados: {len(executed_trades):,}")
        
        # ====================================================================
        # RESUMEN DE RESULTADOS
        # ====================================================================
        print(f"\n  RESULTADOS DE SIMULACIÓN:")
        print(f"  " + "=" * 76)
        
        total = len(opportunities_with_execution)
        profitable = (opportunities_with_execution['profit_category'] == 'Profitable').sum()
        breakeven = (opportunities_with_execution['profit_category'] == 'Break-even').sum()
        loss = (opportunities_with_execution['profit_category'] == 'Loss').sum()
        unknown = (opportunities_with_execution['profit_category'] == 'Unknown').sum()
        
        print(f"    Total oportunidades: {total:,}")
        print(f"    - Profitable: {profitable:,} ({profitable/total*100:.1f}%)")
        print(f"    - Break-even: {breakeven:,} ({breakeven/total*100:.1f}%)")
        print(f"    - Loss: {loss:,} ({loss/total*100:.1f}%)")
        print(f"    - Unknown: {unknown:,} ({unknown/total*100:.1f}%)")
        
        # Profit agregado
        total_theoretical = opportunities_with_execution['total_profit'].sum()
        total_real = opportunities_with_execution['real_total_profit'].sum()
        total_loss = opportunities_with_execution['profit_loss_total'].sum()
        
        print(f"\n  PROFIT AGREGADO:")
        print(f"    - Teórico (latencia=0): €{total_theoretical:.2f}")
        print(f"    - Real (latencia={self.latency_us}us): €{total_real:.2f}")
        print(f"    - Pérdida por latencia: €{total_loss:.2f} ({total_loss/total_theoretical*100:.1f}%)")
        
        # Profit por oportunidad
        if profitable > 0:
            avg_profit_profitable = opportunities_with_execution[
                opportunities_with_execution['profit_category'] == 'Profitable'
            ]['real_total_profit'].mean()
            print(f"    - Profit medio (solo profitable): €{avg_profit_profitable:.4f}")
        
        # REQUISITO 1: Retornar tanto el DataFrame como la lista de trades ejecutados
        return opportunities_with_execution, executed_trades
    
    def sensitivity_analysis(self,
                           signals_df: pd.DataFrame,
                           consolidated_tape: pd.DataFrame,
                           latencies_us: List[int] = None) -> pd.DataFrame:
        """
        Analiza sensibilidad del profit a diferentes latencias.
        
        OBJETIVO:
        Responder: "¿Cuánto profit pierdo si mi latencia aumenta de X a Y?"
        
        Args:
            signals_df: DataFrame con señales detectadas
            consolidated_tape: Tape consolidado
            latencies_us: Lista de latencias a probar (en microsegundos)
                         Default: [10, 50, 100, 200, 500, 1000]
        
        Returns:
            DataFrame con resultados por latencia:
            [latency_us, total_opportunities, profitable_count, 
             total_theoretical_profit, total_real_profit, profit_retention_%]
        """
        
        if latencies_us is None:
            latencies_us = [10, 50, 100, 200, 500, 1000]
        
        print("\n" + "=" * 80)
        print("ANÁLISIS DE SENSIBILIDAD A LATENCIA")
        print("=" * 80)
        print(f"  Latencias a probar: {latencies_us} us")
        
        results = []
        
        for latency in latencies_us:
            print(f"\n  Simulando latencia: {latency} us...")
            
            # Crear simulador temporal
            sim = LatencySimulator(latency_us=latency)
            
            # Ejecutar simulación (REQUISITO 1: ahora retorna tuple)
            exec_df, _ = sim.simulate_execution(signals_df, consolidated_tape, isin=None)
            
            if len(exec_df) == 0:
                continue
            
            # Recopilar métricas
            profitable_count = (exec_df['profit_category'] == 'Profitable').sum()
            total_theoretical = exec_df['total_profit'].sum()
            total_real = exec_df['real_total_profit'].sum()
            retention_pct = (total_real / total_theoretical * 100) if total_theoretical > 0 else 0
            
            results.append({
                'latency_us': latency,
                'total_opportunities': len(exec_df),
                'profitable_count': profitable_count,
                'profitable_pct': profitable_count / len(exec_df) * 100,
                'total_theoretical_profit': total_theoretical,
                'total_real_profit': total_real,
                'profit_loss': total_theoretical - total_real,
                'profit_retention_pct': retention_pct
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n  RESUMEN DE SENSIBILIDAD:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    @staticmethod
    def visualize_latency_impact(sensitivity_df: pd.DataFrame, isin: str):
        """
        Visualiza el impacto de la latencia en el profit.
        
        Args:
            sensitivity_df: DataFrame de sensitivity_analysis()
            isin: ISIN para el título
        """
        
        print("\nGenerando visualizaciones de impacto de latencia...")
        
        if len(sensitivity_df) == 0:
            print("  No hay datos de sensibilidad para visualizar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extraer datos
        latencies = sensitivity_df['latency_us'].values
        
        # ====================================================================
        # Plot 1: Profit Retention vs Latency
        # ====================================================================
        ax1 = axes[0, 0]
        
        ax1.plot(latencies, 
                sensitivity_df['profit_retention_pct'].values,
                marker='o', linewidth=2, markersize=8, color='darkblue')
        
        # Añadir valores en los puntos
        for x, y in zip(latencies, sensitivity_df['profit_retention_pct'].values):
            ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=8)
        
        ax1.set_title('Profit Retention vs Latency', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Latency (us)')
        ax1.set_ylabel('Profit Retention (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # ====================================================================
        # Plot 2: Profitable Opportunities vs Latency
        # ====================================================================
        ax2 = axes[0, 1]
        
        ax2.plot(latencies,
                sensitivity_df['profitable_pct'].values,
                marker='s', linewidth=2, markersize=8, color='darkgreen')
        
        # Añadir valores en los puntos
        for x, y in zip(latencies, sensitivity_df['profitable_pct'].values):
            ax2.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                        xytext=(0,5), ha='center', fontsize=8)
        
        ax2.set_title('Profitable Opportunities vs Latency',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Latency (us)')
        ax2.set_ylabel('Profitable (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # ====================================================================
        # Plot 3: Absolute Profit vs Latency
        # ====================================================================
        ax3 = axes[1, 0]
        
        theoretical = sensitivity_df['total_theoretical_profit'].values
        real = sensitivity_df['total_real_profit'].values
        
        ax3.plot(latencies, theoretical,
                label='Theoretical (latency=0)', 
                marker='o', linewidth=2, color='green', alpha=0.7)
        
        ax3.plot(latencies, real,
                label='Real (with latency)',
                marker='s', linewidth=2, color='red', alpha=0.7)
        
        ax3.set_title('Absolute Profit vs Latency',
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Latency (us)')
        ax3.set_ylabel('Total Profit (€)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # ====================================================================
        # Plot 4: Profit Loss vs Latency
        # ====================================================================
        ax4 = axes[1, 1]
        
        loss_values = sensitivity_df['profit_loss'].values
        x_positions = range(len(sensitivity_df))
        
        bars = ax4.bar(x_positions, loss_values, color='coral', alpha=0.7)
        
        # Añadir valores encima de las barras
        for i, (bar, val) in enumerate(zip(bars, loss_values)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'€{val:.2f}',
                    ha='center', va='bottom', fontsize=8)
        
        ax4.set_title('Profit Loss vs Latency',
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Latency Configuration')
        ax4.set_ylabel('Profit Loss (€)')
        ax4.set_xticks(x_positions)
        ax4.set_xticklabels([f"{lat}us" for lat in latencies], 
                           rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Latency Impact Analysis - ISIN: {isin}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Guardar y mostrar
        output_path = config.FIGURES_DIR / f'latency_impact_{isin}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Visualización guardada: {output_path}")
        
        try:
            plt.show(block=False)
            plt.pause(0.5)  # Pausa más larga para asegurar renderizado
            print(f"  [OK] Gráficas mostradas en ventana")
        except Exception as e:
            logger.warning(f"  No se pudo mostrar gráficas interactivas: {e}")
            print(f"  [INFO] Gráficas guardadas en: {output_path}")
        
        # No cerrar inmediatamente para que el usuario pueda verlas
        # plt.close()  # Comentado para permitir visualización
    
    @staticmethod
    def export_execution_results(exec_df: pd.DataFrame, 
                                output_path: str = None):
        """
        Exporta resultados de ejecución con latencia a CSV.
        
        Args:
            exec_df: DataFrame de simulate_execution()
            output_path: Path para guardar (opcional)
        """
        
        if len(exec_df) == 0:
            print("  No hay resultados de ejecución para exportar")
            return
        
        if output_path is None:
            output_path = config.OUTPUT_DIR / "execution_results.csv"
        
        # Seleccionar columnas relevantes
        cols_to_export = [
            'epoch', 'execution_epoch',
            'venue_max_bid', 'venue_min_ask',
            'global_max_bid', 'global_min_ask',
            'execution_bid', 'execution_ask',
            'theoretical_profit', 'real_profit', 'profit_loss',
            'executable_qty', 'total_profit', 'real_total_profit',
            'profit_category', 'is_profitable'
        ]
        
        available_cols = [col for col in cols_to_export if col in exec_df.columns]
        
        exec_df[available_cols].to_csv(output_path, index=False)
        
        print(f"\n  Resultados de ejecución exportados: {output_path}")
        print(f"    Total filas: {len(exec_df):,}")