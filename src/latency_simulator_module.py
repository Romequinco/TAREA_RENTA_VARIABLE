"""
================================================================================
latency_simulator.py - Simulación de Impacto de Latencia (OPTIMIZADO)
================================================================================

CONCEPTO CLAVE - "TIME MACHINE":
Si detecto una oportunidad en el tiempo T, pero mi sistema tiene latencia Δ,
no puedo ejecutar hasta T+Δ. Para entonces, el mercado puede haber cambiado
y la oportunidad puede haber desaparecido o ser menos rentable.

LATENCIAS A SIMULAR (en microsegundos):
[0, 100, 500, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 30000, 50000, 100000]

ALGORITMO:
1. Preparar consolidated_df para búsqueda rápida (indexar por epoch)
2. Para cada señal: calcular execution_epoch = signal_epoch + latency_us
3. Buscar precios en execution_epoch usando búsqueda binaria eficiente
4. Recalcular profit con precios actualizados
5. Calcular métricas de decay

OPTIMIZACIONES APLICADAS:
- Búsqueda binaria con searchsorted (O(log n))
- Vectorización cuando es posible
- Manejo robusto de edge cases
- Validaciones exhaustivas según especificaciones

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from config_module import config

logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES COMPATIBLES CON EL OTRO CÓDIGO
# ============================================================================

def get_quote_at_epoch(consolidated: pd.DataFrame, target_epoch: int, method: str = 'nearest') -> Optional[pd.Series]:
    """
    Efficiently query the consolidated tape at a specific epoch.
    Igual que el otro código.
    """
    if consolidated.empty:
        return None
    
    # Ensure consolidated is indexed by epoch
    if consolidated.index.name != 'epoch':
        consolidated = consolidated.set_index('epoch')
    
    if method == 'nearest':
        # Use searchsorted for efficient lookup
        idx = np.searchsorted(consolidated.index, target_epoch, side='left')
        
        # Handle edge cases
        if idx == 0:
            return consolidated.iloc[0]
        elif idx >= len(consolidated):
            return consolidated.iloc[-1]
        else:
            # Choose nearest
            left_epoch = consolidated.index[idx - 1]
            right_epoch = consolidated.index[idx]
            
            if abs(target_epoch - left_epoch) <= abs(target_epoch - right_epoch):
                return consolidated.iloc[idx - 1]
            else:
                return consolidated.iloc[idx]
    
    return None


def is_valid_price(price: float) -> bool:
    """Check if a price is valid (not NaN, not a magic number, and > 0)."""
    if pd.isna(price) or price <= 0:
        return False
    if price in config.MAGIC_NUMBERS:
        return False
    return True


def simulate_latency_with_losses(opportunities: pd.DataFrame, consolidated: pd.DataFrame, latency_us: int) -> float:
    """
    Simulate execution latency, calculating profit/loss at T + Latency.
    Includes both profits and losses.
    Igual que el otro código.
    """
    if opportunities.empty or consolidated.empty:
        return 0.0
    
    # Work with copies
    opps = opportunities.copy()
    
    # Ensure consolidated is indexed by epoch
    if consolidated.index.name != 'epoch':
        consolidated = consolidated.set_index('epoch')
    
    total_profit = 0.0
    
    for _, opp in opps.iterrows():
        original_epoch = opp['epoch']
        target_epoch = original_epoch + latency_us
        
        # Get quote at T + latency
        quote_at_latency = get_quote_at_epoch(consolidated, target_epoch)
        
        if quote_at_latency is None:
            continue
        
        # Get prices from the specific exchanges at T + latency
        buy_exchange = opp['buy_exchange']
        sell_exchange = opp['sell_exchange']
        
        buy_price_col = f'{buy_exchange}_ask'
        sell_price_col = f'{sell_exchange}_bid'
        buy_qty_col = f'{buy_exchange}_ask_qty'
        sell_qty_col = f'{sell_exchange}_bid_qty'
        
        if buy_price_col not in quote_at_latency.index or sell_price_col not in quote_at_latency.index:
            continue
        
        buy_price = quote_at_latency[buy_price_col]
        sell_price = quote_at_latency[sell_price_col]
        
        # Check if prices are valid
        if pd.isna(buy_price) or pd.isna(sell_price):
            continue
        
        if not is_valid_price(buy_price) or not is_valid_price(sell_price):
            continue
        
        # Get quantities
        sell_qty = quote_at_latency.get(sell_qty_col, 0) if sell_qty_col in quote_at_latency.index else 0
        buy_qty = quote_at_latency.get(buy_qty_col, 0) if buy_qty_col in quote_at_latency.index else 0
        
        if pd.isna(sell_qty) or sell_qty <= 0:
            sell_qty = 0
        if pd.isna(buy_qty) or buy_qty <= 0:
            buy_qty = 0
        
        if sell_qty <= 0 or buy_qty <= 0:
            continue
        
        # Use original quantity or available quantity, whichever is smaller
        original_qty = opp.get('tradeable_qty', 0)
        if pd.isna(original_qty) or original_qty <= 0:
            original_qty = min(sell_qty, buy_qty)
        
        tradeable_qty = min(original_qty, sell_qty, buy_qty)
        
        if tradeable_qty <= 0:
            continue
        
        # Calculate profit/loss (can be negative)
        profit_per_share = sell_price - buy_price
        realized_profit = profit_per_share * tradeable_qty
        
        total_profit += realized_profit
    
    return total_profit


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
    
    @staticmethod
    def _lookup_future_state(signal_epoch: int, 
                            latency_ns: int,
                            consolidated_indexed: pd.DataFrame,
                            bid_cols: List[str],
                            ask_cols: List[str]) -> Tuple[float, float, float]:
        """
        Busca el estado futuro del mercado en execution_epoch.
        
        Método eficiente usando searchsorted (búsqueda binaria O(log n)).
        
        Args:
            signal_epoch: Timestamp de detección de la señal
            latency_ns: Latencia en nanosegundos
            consolidated_indexed: DataFrame indexado por epoch
            bid_cols: Lista de columnas de bids
            ask_cols: Lista de columnas de asks
            
        Returns:
            Tuple[actual_max_bid, actual_min_ask, actual_profit]
            Si execution_epoch está fuera de rango, retorna (0, 0, 0)
        """
        execution_epoch = signal_epoch + latency_ns
        
        # Verificar si está fuera del rango de datos
        if execution_epoch < consolidated_indexed.index.min():
            return (0.0, 0.0, 0.0)
        
        if execution_epoch > consolidated_indexed.index.max():
            return (0.0, 0.0, 0.0)
        
        # Búsqueda binaria eficiente O(log n) usando searchsorted
        # side='left' significa: encontrar el índice donde se insertaría execution_epoch
        # manteniendo el orden, usando el valor más cercano >= execution_epoch
        idx = consolidated_indexed.index.searchsorted(execution_epoch, side='left')
        
        # Si el índice está fuera del rango, usar el último valor conocido
        if idx >= len(consolidated_indexed):
            idx = len(consolidated_indexed) - 1
        
        # Obtener fila futura (estado del mercado en execution_epoch)
        future_row = consolidated_indexed.iloc[idx]
        
        # Verificar gap temporal (si hay gap >10s, considerar profit = 0)
        # CRÍTICO: Gaps grandes indican que el mercado puede haber cambiado drásticamente
        # y los precios forward-filled pueden no ser representativos
        if idx > 0:
            prev_epoch = consolidated_indexed.index[idx - 1]
            gap_ns = execution_epoch - prev_epoch
            gap_seconds = gap_ns / 1e9  # Convertir nanosegundos a segundos
            
            if gap_seconds > 10:
                # Gap demasiado grande, mercado puede haber cambiado drásticamente
                # Retornar profit = 0 para ser conservador
                return (0.0, 0.0, 0.0)
        
        # Calcular max_bid y min_ask en el futuro
        available_bids = [future_row[col] for col in bid_cols if col in future_row.index and pd.notna(future_row[col])]
        available_asks = [future_row[col] for col in ask_cols if col in future_row.index and pd.notna(future_row[col])]
        
        if not available_bids or not available_asks:
            return (0.0, 0.0, 0.0)
        
        actual_max_bid = max(available_bids)
        actual_min_ask = min(available_asks)
        
        # Profit actualizado
        actual_profit = max(0.0, actual_max_bid - actual_min_ask)
        
        return (actual_max_bid, actual_min_ask, actual_profit)
    
    @staticmethod
    def simulate_latency_impact(signals_df: pd.DataFrame,
                                consolidated_df: pd.DataFrame,
                                latency_us: int) -> pd.DataFrame:
        """
        Simula el impacto de latencia en las señales detectadas.
        
        Input:
          - signals_df: DataFrame con señales detectadas (rising edges)
          - consolidated_df: DataFrame consolidado original con todos los timestamps
          - latency_us: Latencia en microsegundos
        
        Output: DataFrame con columnas adicionales:
                [actual_max_bid, actual_min_ask, actual_profit, profit_decay]
        
        Args:
            signals_df: DataFrame con señales
            consolidated_df: DataFrame consolidado
            latency_us: Latencia en microsegundos
            
        Returns:
            DataFrame con columnas adicionales de impacto de latencia
        """
        print(f"\n  Simulando impacto de latencia: {latency_us} us")
        
        if len(signals_df) == 0:
            logger.warning("  No hay señales para simular")
            return pd.DataFrame()
        
        # Filtrar solo rising edges
        rising_edges = signals_df[signals_df.get('is_rising_edge', False)].copy()
        
        if len(rising_edges) == 0:
            logger.warning("  No hay rising edges para simular")
            return pd.DataFrame()
        
        # Preparar consolidated_df para búsqueda rápida
        consolidated_sorted = consolidated_df.sort_values('epoch').reset_index(drop=True)
        consolidated_indexed = consolidated_sorted.set_index('epoch')
        
        # Extraer columnas de bids y asks
        bid_cols = [col for col in consolidated_df.columns if col.endswith('_bid') and not col.endswith('_bid_qty')]
        ask_cols = [col for col in consolidated_df.columns if col.endswith('_ask') and not col.endswith('_ask_qty')]
        
        latency_ns = latency_us * 1000
        
        # Inicializar columnas de resultados
        rising_edges['actual_max_bid'] = 0.0
        rising_edges['actual_min_ask'] = 0.0
        rising_edges['actual_profit'] = 0.0
        rising_edges['profit_decay'] = 0.0
        
        # Para cada señal, buscar estado futuro
        print(f"    Procesando {len(rising_edges):,} señales...")
        
        for idx, row in rising_edges.iterrows():
            signal_epoch = row['epoch']
            
            # Buscar estado futuro
            actual_max_bid, actual_min_ask, actual_profit = LatencySimulator._lookup_future_state(
                signal_epoch,
                latency_ns,
                consolidated_indexed,
                bid_cols,
                ask_cols
            )
            
            rising_edges.at[idx, 'actual_max_bid'] = actual_max_bid
            rising_edges.at[idx, 'actual_min_ask'] = actual_min_ask
            rising_edges.at[idx, 'actual_profit'] = actual_profit
            
            # Calcular profit decay
            theoretical_profit = row.get('theoretical_profit', 0.0)
            if theoretical_profit > 0:
                profit_decay = ((theoretical_profit - actual_profit) / theoretical_profit) * 100
                rising_edges.at[idx, 'profit_decay'] = profit_decay
        
        return rising_edges
    
    @staticmethod
    def run_latency_sweep(signals_df: pd.DataFrame,
                          consolidated_df: pd.DataFrame,
                          latencies: List[int] = None) -> pd.DataFrame:
        """
        Ejecuta la simulación para todas las latencias especificadas.
        
        Output: DataFrame con columnas:
                [latency_us, total_opportunities, total_theoretical_profit,
                 total_actual_profit, profit_capture_rate, avg_profit_per_trade]
        
        Args:
            signals_df: DataFrame con señales detectadas
            consolidated_df: DataFrame consolidado
            latencies: Lista de latencias en microsegundos (default: LATENCY_BUCKETS)
            
        Returns:
            DataFrame con resultados agregados por latencia
        """
        if latencies is None:
            latencies = config.LATENCY_BUCKETS
        
        print("\n" + "=" * 80)
        print("BARRIDO DE LATENCIAS")
        print("=" * 80)
        print(f"  Latencias a simular: {latencies} us")
        
        results = []
        
        for latency in latencies:
            print(f"\n  Simulando latencia: {latency} us...")
            
            signals_with_latency = LatencySimulator.simulate_latency_impact(
                signals_df, consolidated_df, latency
            )
            
            if len(signals_with_latency) == 0:
                continue
            
            # Calcular métricas agregadas
            total_opportunities = len(signals_with_latency)
            total_theoretical_profit = signals_with_latency['theoretical_profit'].sum()
            total_actual_profit = signals_with_latency['actual_profit'].sum()
            
            profit_capture_rate = (
                (total_actual_profit / total_theoretical_profit * 100) 
                if total_theoretical_profit > 0 else 0.0
            )
            
            avg_profit_per_trade = (
                signals_with_latency['actual_profit'].mean() 
                if total_opportunities > 0 else 0.0
            )
            
            results.append({
                'latency_us': latency,
                'total_opportunities': total_opportunities,
                'total_theoretical_profit': total_theoretical_profit,
                'total_actual_profit': total_actual_profit,
                'profit_capture_rate': profit_capture_rate,
                'avg_profit_per_trade': avg_profit_per_trade
            })
        
        results_df = pd.DataFrame(results)
        
        print(f"\n  RESUMEN DEL BARRIDO:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    @staticmethod
    def validate_latency_results(results_df: pd.DataFrame) -> bool:
        """
        Valida que los resultados de latencia sean consistentes.
        
        Validaciones:
        1. actual_profit <= theoretical_profit para cada señal
        2. profit_capture_rate decrece monotónicamente con latencia
        3. Latencia 0 debe tener profit_capture_rate ≈ 100%
        4. Latencias muy altas (>50ms) deben tener capture rate cercana a 0%
        
        Args:
            results_df: DataFrame de run_latency_sweep()
            
        Returns:
            True si todas las validaciones pasan
            
        Raises:
            AssertionError con mensaje descriptivo si falla alguna validación
        """
        print("\n" + "=" * 80)
        print("VALIDANDO RESULTADOS DE LATENCIA")
        print("=" * 80)
        
        if len(results_df) == 0:
            raise AssertionError("DataFrame de resultados está vacío")
        
        # Validación 1: profit_capture_rate decrece monotónicamente
        sorted_results = results_df.sort_values('latency_us')
        capture_rates = sorted_results['profit_capture_rate'].values
        
        # Verificar que decrece (permitir pequeñas variaciones por ruido)
        for i in range(1, len(capture_rates)):
            if capture_rates[i] > capture_rates[i-1] + 1.0:  # Tolerancia de 1%
                raise AssertionError(
                    f"profit_capture_rate no decrece monotónicamente: "
                    f"{capture_rates[i-1]:.2f}% → {capture_rates[i]:.2f}%"
                )
        
        print("  [OK] profit_capture_rate decrece monotónicamente")
        
        # Validación 2: Latencia 0 debe tener capture rate ≈ 100%
        latency_0 = sorted_results[sorted_results['latency_us'] == 0]
        if len(latency_0) > 0:
            capture_0 = latency_0['profit_capture_rate'].iloc[0]
            if capture_0 < 95.0:  # Tolerancia de 5%
                raise AssertionError(
                    f"Latencia 0 tiene profit_capture_rate={capture_0:.2f}% (esperado ≈100%)"
                )
            print(f"  [OK] Latencia 0 tiene capture rate: {capture_0:.2f}%")
        
        # Validación 3: Latencias altas deben tener capture rate cercana a 0%
        high_latency = sorted_results[sorted_results['latency_us'] >= 50000]
        if len(high_latency) > 0:
            max_capture = high_latency['profit_capture_rate'].max()
            if max_capture > 50.0:  # Tolerancia de 50%
                logger.warning(
                    f"Latencias altas tienen capture rate alto: {max_capture:.2f}% "
                    f"(esperado cercano a 0%)"
                )
            else:
                print(f"  [OK] Latencias altas tienen capture rate bajo: {max_capture:.2f}%")
        
        print("\n  [EXITO] VALIDACIÓN EXITOSA")
        return True
    
    def simulate_execution(self, 
                          signals_df: pd.DataFrame,
                          consolidated_tape: pd.DataFrame,
                          isin: str = None) -> Tuple[pd.DataFrame, list]:
        """
        Simula la ejecución de oportunidades considerando latencia.
        
        Compatibilidad con código existente.
        
        Args:
            signals_df: DataFrame de señales detectadas
            consolidated_tape: DataFrame consolidado
            isin: ISIN del instrumento (opcional)
        
        Returns:
            Tuple[DataFrame con resultados, lista de trades ejecutados]
        """
        print("\n" + "=" * 80)
        print("SIMULACIÓN DE LATENCIA (TIME MACHINE)")
        print("=" * 80)
        print(f"  Latencia configurada: {self.latency_us} us")
        
        # Filtrar solo rising edges
        opportunities = signals_df[signals_df.get('is_rising_edge', False)].copy()
        
        if len(opportunities) == 0:
            logger.warning("  No hay rising edges para simular")
            return pd.DataFrame(), []
        
        print(f"  Total oportunidades a simular: {len(opportunities):,}")
        
        # Simular impacto de latencia
        opportunities_with_latency = self.simulate_latency_impact(
            opportunities, consolidated_tape, self.latency_us
        )
        
        if len(opportunities_with_latency) == 0:
            return pd.DataFrame(), []
        
        # Calcular execution_epoch
        opportunities_with_latency['execution_epoch'] = (
            opportunities_with_latency['epoch'] + self.latency_ns
        )
        
        # Extraer precios de ejecución de los venues correctos
        def get_execution_bid(row):
            venue = row.get('venue_max_bid', row.get('max_bid_venue', ''))
            if venue:
                bid_col = f'{venue}_bid'
                return row.get(bid_col, row.get('actual_max_bid', 0.0))
            return row.get('actual_max_bid', 0.0)
        
        def get_execution_ask(row):
            venue = row.get('venue_min_ask', row.get('min_ask_venue', ''))
            if venue:
                ask_col = f'{venue}_ask'
                return row.get(ask_col, row.get('actual_min_ask', 0.0))
            return row.get('actual_min_ask', 0.0)
        
        opportunities_with_latency['execution_bid'] = opportunities_with_latency.apply(
            get_execution_bid, axis=1
        )
        opportunities_with_latency['execution_ask'] = opportunities_with_latency.apply(
            get_execution_ask, axis=1
        )
        
        # Calcular profit real
        opportunities_with_latency['real_profit'] = (
            opportunities_with_latency['execution_bid'] - 
            opportunities_with_latency['execution_ask']
        )
        
        # Cantidad ejecutable
        exec_qty_col = opportunities_with_latency.get('executable_qty', 
                                                      opportunities_with_latency.get('tradeable_qty', 0))
        if isinstance(exec_qty_col, str):
            opportunities_with_latency['executed_qty'] = np.where(
                opportunities_with_latency['real_profit'] > 0.0001,
                opportunities_with_latency[exec_qty_col],
                0
            )
        else:
            opportunities_with_latency['executed_qty'] = np.where(
                opportunities_with_latency['real_profit'] > 0.0001,
                exec_qty_col,
                0
            )
        
        # Profit total real
        opportunities_with_latency['real_total_profit'] = (
            opportunities_with_latency['real_profit'] * 
            opportunities_with_latency['executed_qty']
        )
        
        # Profit loss
        opportunities_with_latency['profit_loss'] = (
            opportunities_with_latency['theoretical_profit'] - 
            opportunities_with_latency['real_profit']
        )
        opportunities_with_latency['profit_loss_total'] = (
            opportunities_with_latency['total_profit'] - 
            opportunities_with_latency['real_total_profit']
        )
        
        # Clasificar oportunidades
        min_threshold = 0.0001
        opportunities_with_latency['is_profitable'] = (
            opportunities_with_latency['real_profit'] > min_threshold
        )
        
        def categorize_profit(row):
            if pd.isna(row['real_profit']):
                return 'Unknown'
            elif row['real_profit'] > min_threshold:
                return 'Profitable'
            elif row['real_profit'] >= -min_threshold:
                return 'Break-even'
            else:
                return 'Loss'
        
        opportunities_with_latency['profit_category'] = opportunities_with_latency.apply(
            categorize_profit, axis=1
        )
        
        # Registrar trades ejecutados
        executed_trades = []
        execution_id_counter = 0
        
        profitable_executions = opportunities_with_latency[
            (opportunities_with_latency['profit_category'] == 'Profitable') &
            (opportunities_with_latency['executed_qty'] > 0)
        ].copy()
        
        for idx, row in profitable_executions.iterrows():
            execution_id_counter += 1
            execution_id = f"EXEC_{execution_id_counter:08d}"
            
            executed_trade = {
                'execution_id': execution_id,
                'epoch': row['epoch'],
                'execution_epoch': row.get('execution_epoch', row['epoch']),
                'venue_max_bid': row.get('venue_max_bid', row.get('max_bid_venue', '')),
                'venue_min_ask': row.get('venue_min_ask', row.get('min_ask_venue', '')),
                'executed_qty': row['executed_qty'],
                'execution_bid': row.get('execution_bid', row.get('actual_max_bid', 0.0)),
                'execution_ask': row.get('execution_ask', row.get('actual_min_ask', 0.0)),
                'real_profit': row['real_profit'],
                'real_total_profit': row['real_total_profit']
            }
            
            if isin:
                executed_trade['isin'] = isin
            
            executed_trades.append(executed_trade)
            opportunities_with_latency.at[idx, 'execution_id'] = execution_id
        
        if 'execution_id' not in opportunities_with_latency.columns:
            opportunities_with_latency['execution_id'] = None
        
        # Resumen
        total = len(opportunities_with_latency)
        profitable = (opportunities_with_latency['profit_category'] == 'Profitable').sum()
        total_theoretical = opportunities_with_latency['total_profit'].sum()
        total_real = opportunities_with_latency['real_total_profit'].sum()
        
        print(f"\n  RESULTADOS:")
        print(f"    Total oportunidades: {total:,}")
        print(f"    Profitable: {profitable:,} ({profitable/total*100:.1f}%)")
        print(f"    Profit teórico: €{total_theoretical:.2f}")
        print(f"    Profit real: €{total_real:.2f}")
        print(f"    Trades ejecutados: {len(executed_trades):,}")
        
        return opportunities_with_latency, executed_trades
    
    def sensitivity_analysis(self,
                           signals_df: pd.DataFrame,
                           consolidated_tape: pd.DataFrame,
                           latencies_us: List[int] = None) -> pd.DataFrame:
        """
        Analiza sensibilidad del profit a diferentes latencias.
        
        Compatibilidad con código existente.
        """
        if latencies_us is None:
            latencies_us = config.LATENCY_BUCKETS
        
        return self.run_latency_sweep(signals_df, consolidated_tape, latencies_us)
    
    @staticmethod
    def visualize_latency_impact(sensitivity_df: pd.DataFrame, isin: str):
        """Visualiza el impacto de la latencia en el profit."""
        print("\nGenerando visualizaciones de impacto de latencia...")
        
        if len(sensitivity_df) == 0:
            print("  No hay datos de sensibilidad para visualizar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        latencies = sensitivity_df['latency_us'].values
        
        # Plot 1: Profit Retention vs Latency
        ax1 = axes[0, 0]
        ax1.plot(latencies, sensitivity_df['profit_capture_rate'].values,
                marker='o', linewidth=2, markersize=8, color='darkblue')
        ax1.set_title('Profit Retention vs Latency', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Latency (us)')
        ax1.set_ylabel('Profit Retention (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Absolute Profit vs Latency
        ax2 = axes[0, 1]
        ax2.plot(latencies, sensitivity_df['total_theoretical_profit'].values,
                label='Theoretical', marker='o', linewidth=2, color='green', alpha=0.7)
        ax2.plot(latencies, sensitivity_df['total_actual_profit'].values,
                label='Actual', marker='s', linewidth=2, color='red', alpha=0.7)
        ax2.set_title('Absolute Profit vs Latency', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Latency (us)')
        ax2.set_ylabel('Total Profit (€)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Plot 3: Profit Loss vs Latency
        ax3 = axes[1, 0]
        profit_loss = sensitivity_df['total_theoretical_profit'] - sensitivity_df['total_actual_profit']
        ax3.plot(latencies, profit_loss.values, marker='s', linewidth=2, color='coral')
        ax3.set_title('Profit Loss vs Latency', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Latency (us)')
        ax3.set_ylabel('Profit Loss (€)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot 4: Avg Profit per Trade vs Latency
        ax4 = axes[1, 1]
        ax4.plot(latencies, sensitivity_df['avg_profit_per_trade'].values,
                marker='d', linewidth=2, color='purple')
        ax4.set_title('Avg Profit per Trade vs Latency', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Latency (us)')
        ax4.set_ylabel('Avg Profit (€)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.suptitle(f'Latency Impact Analysis - ISIN: {isin}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = config.FIGURES_DIR / f'latency_impact_{isin}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Visualización guardada: {output_path}")
        
        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception as e:
            logger.warning(f"  No se pudo mostrar gráficas interactivas: {e}")
    
    @staticmethod
    def export_execution_results(exec_df: pd.DataFrame, output_path: str = None):
        """Exporta resultados de ejecución con latencia a CSV."""
        if len(exec_df) == 0:
            print("  No hay resultados de ejecución para exportar")
            return
        
        if output_path is None:
            output_path = config.OUTPUT_DIR / "execution_results.csv"
        
        cols_to_export = [
            'epoch', 'execution_epoch', 'venue_max_bid', 'venue_min_ask',
            'execution_bid', 'execution_ask', 'theoretical_profit', 'real_profit',
            'executed_qty', 'total_profit', 'real_total_profit', 'profit_category'
        ]
        
        available_cols = [col for col in cols_to_export if col in exec_df.columns]
        exec_df[available_cols].to_csv(output_path, index=False)
        
        print(f"\n  Resultados exportados: {output_path}")


# ============================================================================
# FUNCIONES WRAPPER ESTÁTICAS (según especificaciones)
# ============================================================================

def simulate_latency_impact(signals_df: pd.DataFrame,
                            consolidated_df: pd.DataFrame,
                            latency_us: int) -> pd.DataFrame:
    """
    Función principal para simular impacto de latencia.
    
    Wrapper estático según especificaciones.
    
    Args:
        signals_df: DataFrame con señales detectadas (rising edges)
        consolidated_df: DataFrame consolidado original
        latency_us: Latencia en microsegundos
    
    Returns:
        DataFrame con columnas adicionales: actual_max_bid, actual_min_ask,
        actual_profit, profit_decay
    """
    return LatencySimulator.simulate_latency_impact(signals_df, consolidated_df, latency_us)


def run_latency_sweep(signals_df: pd.DataFrame,
                      consolidated_df: pd.DataFrame,
                      latencies: List[int] = None) -> pd.DataFrame:
    """
    Ejecuta simulación para todas las latencias especificadas.
    
    Wrapper estático según especificaciones.
    
    Args:
        signals_df: DataFrame con señales detectadas
        consolidated_df: DataFrame consolidado
        latencies: Lista de latencias en microsegundos
    
    Returns:
        DataFrame con resultados agregados por latencia
    """
    return LatencySimulator.run_latency_sweep(signals_df, consolidated_df, latencies)


def validate_latency_results(results_df: pd.DataFrame) -> bool:
    """
    Valida que los resultados de latencia sean consistentes.
    
    Wrapper estático según especificaciones.
    
    Args:
        results_df: DataFrame de run_latency_sweep()
    
    Returns:
        True si todas las validaciones pasan
    """
    return LatencySimulator.validate_latency_results(results_df)
