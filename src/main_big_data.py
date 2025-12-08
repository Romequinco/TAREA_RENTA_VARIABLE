"""
================================================================================
main_big.py - Script Principal para DATA_BIG (PRODUCCIÓN)
================================================================================

Este script procesa el dataset completo (DATA_BIG) con optimizaciones adicionales:
- Procesamiento por chunks si es necesario
- Mayor time_bin para reducir memoria
- Análisis por múltiples ISINs
- Generación de reportes consolidados

DIFERENCIAS vs main.py (DATA_SMALL):
- Usa DATA_BIG_DIR en lugar de DATA_SMALL_DIR
- time_bin_ms = 200 (más agresivo para reducir memoria)
- Procesa TODOS los ISINs disponibles (o los primeros N)
- Genera reporte agregado multi-ISIN

Uso:
    python main_big.py --max-isins 5 --time-bin 200

================================================================================
"""

import logging
import sys
import argparse
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict

# Verificar si pandas tiene to_markdown (requiere tabulate)
try:
    _test_df = pd.DataFrame({'a': [1]})
    _test_df.to_markdown()
    HAS_MARKDOWN = True
except:
    HAS_MARKDOWN = False
    try:
        import tabulate
    except ImportError:
        print("  [ADVERTENCIA] tabulate no está instalado. Las tablas se mostrarán en formato texto.")

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Importar módulos del proyecto
from config_module import config
from data_loader_module import DataLoader
from data_cleaner_module import DataCleaner
from consolidator_module import ConsolidatedTape
from signal_generator_module import SignalGenerator
from latency_simulator_module import LatencySimulator
from analyzer_module import ArbitrageAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_DIR / 'arbitrage_system_BIG.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def clean_output_directories():
    """
    Limpia todos los archivos de output antes de una nueva ejecución.
    Elimina archivos en OUTPUT_DIR y FIGURES_DIR pero mantiene los directorios.
    """
    print("\n" + "=" * 80)
    print("LIMPIEZA DE OUTPUTS ANTERIORES")
    print("=" * 80)
    
    cleaned_files = 0
    
    # Limpiar archivos en OUTPUT_DIR (excepto el directorio mismo)
    if config.OUTPUT_DIR.exists():
        for item in config.OUTPUT_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    cleaned_files += 1
                elif item.is_dir() and item.name == 'figures':
                    # Limpiar contenido de figures pero mantener el directorio
                    for fig_file in item.iterdir():
                        if fig_file.is_file():
                            fig_file.unlink()
                            cleaned_files += 1
            except Exception as e:
                logger.warning(f"  No se pudo eliminar {item}: {e}")
    
    # Asegurar que los directorios existen
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    print(f"  Archivos eliminados: {cleaned_files}")
    print(f"  [OK] Directorios de output listos para nueva ejecución")
    print("=" * 80)


def display_dataframe(df: pd.DataFrame, title: str = "", max_rows: int = 20):
    """
    Muestra un DataFrame de forma clara y legible.
    
    Args:
        df: DataFrame a mostrar
        title: Título para la tabla
        max_rows: Número máximo de filas a mostrar
    """
    if df is None or len(df) == 0:
        print(f"\n  {title}: (vacío)")
        return
    
    print(f"\n  {'=' * 76}")
    if title:
        print(f"  {title}")
        print(f"  {'=' * 76}")
    
    # Mostrar solo las primeras max_rows filas
    display_df = df.head(max_rows)
    
    # Configurar pandas para mostrar más columnas
    with pd.option_context('display.max_columns', None,
                          'display.width', None,
                          'display.max_colwidth', 50):
        print(display_df.to_string(index=False))
    
    if len(df) > max_rows:
        print(f"\n  ... (mostrando {max_rows} de {len(df)} filas totales)")
    
    print(f"  {'=' * 76}")


def process_single_isin(isin: str, 
                       loader: DataLoader, 
                       time_bin_ms: int = 200) -> dict:
    """
    Procesa un ISIN completo (todas las fases del pipeline).
    
    Args:
        isin: ISIN a procesar
        loader: DataLoader inicializado
        time_bin_ms: Ventana temporal para consolidación
    
    Returns:
        Dict con resultados del análisis
    """
    
    print("\n" + "=" * 80)
    print(f"PROCESANDO ISIN: {isin}")
    print("=" * 80)
    
    results = {
        'isin': isin,
        'success': False,
        'error': None,
        'metrics': None
    }
    
    try:
        # ====================================================================
        # FASE 1: CARGA DE DATOS
        # ====================================================================
        print(f"\n[FASE 1] Cargando datos para {isin}...")
        raw_data = loader.load_isin_data(isin)
        
        if len(raw_data) == 0:
            results['error'] = "No data loaded"
            logger.warning(f"  [SKIP] {isin} - No se pudieron cargar datos")
            return results
        
        # ====================================================================
        # FASE 2: LIMPIEZA
        # ====================================================================
        print(f"\n[FASE 2] Limpiando datos para {isin}...")
        cleaner = DataCleaner()
        clean_data = cleaner.clean_all_venues(raw_data)
        
        if len(clean_data) == 0:
            results['error'] = "No data after cleaning"
            logger.warning(f"  [SKIP] {isin} - No quedan datos después de limpieza")
            return results
        
        # ====================================================================
        # FASE 3: CONSOLIDATED TAPE
        # ====================================================================
        print(f"\n[FASE 3] Creando consolidated tape para {isin}...")
        tape_builder = ConsolidatedTape(time_bin_ms=time_bin_ms)
        
        consolidated_tape = tape_builder.create_tape(clean_data)
        
        if consolidated_tape is None or len(consolidated_tape) == 0:
            results['error'] = "Consolidated tape empty"
            logger.warning(f"  [SKIP] {isin} - Tape consolidado vacío")
            return results
        
        # Validar tape
        is_valid = tape_builder.validate_tape(consolidated_tape)
        if not is_valid:
            results['error'] = "Tape validation failed"
            logger.warning(f"  [SKIP] {isin} - Validación de tape falló")
            return results
        
        # Estadísticas del tape (sin visualización individual)
        print("\n  [ESTADÍSTICAS DEL CONSOLIDATED TAPE]")
        tape_stats = tape_builder.compute_statistics(consolidated_tape)
        # NO mostrar visualización individual - se harán gráficas agregadas al final
        
        # ====================================================================
        # FASE 4: DETECCIÓN DE SEÑALES
        # ====================================================================
        print(f"\n[FASE 4] Detectando señales para {isin}...")
        signal_gen = SignalGenerator()
        
        # REQUISITO 1: Inicializar lista de trades ejecutados (vacía al inicio)
        executed_trades = []
        
        # Primera detección sin trades ejecutados
        signals_df = signal_gen.detect_opportunities(
            consolidated_tape, 
            executed_trades=executed_trades,
            isin=isin
        )
        
        # Mostrar resumen de oportunidades detectadas
        if signals_df is not None and len(signals_df) > 0:
            rising_edges = signals_df[signals_df['is_rising_edge']].copy()
            if len(rising_edges) > 0:
                print(f"\n  [RESUMEN DE OPORTUNIDADES DETECTADAS]")
                summary_cols = ['epoch', 'venue_max_bid', 'venue_min_ask', 
                              'executable_qty', 'theoretical_profit', 'total_profit']
                available_summary_cols = [c for c in summary_cols if c in rising_edges.columns]
                display_dataframe(
                    rising_edges[available_summary_cols].head(10),
                    f"Primeras 10 Oportunidades (de {len(rising_edges)} totales)",
                    max_rows=10
                )
        
        print("\n  [ANÁLISIS DE PARES DE VENUES]")
        venue_pairs = signal_gen.analyze_venue_pairs(signals_df)
        # NO mostrar visualización individual - se harán gráficas agregadas al final
        
        print("\n  [EXPORTANDO OPORTUNIDADES]")
        signal_gen.export_opportunities(
            signals_df, 
            output_path=config.OUTPUT_DIR / f"opportunities_{isin}.csv"
        )
        
        # ====================================================================
        # FASE 5: SIMULACIÓN DE LATENCIA
        # ====================================================================
        print(f"\n[FASE 5] Simulando latencia para {isin}...")
        sim = LatencySimulator(latency_us=100)
        # REQUISITO 1: simulate_execution ahora retorna (DataFrame, executed_trades_list)
        try:
            result = sim.simulate_execution(
                signals_df, 
                consolidated_tape,
                isin=isin
            )
            # Manejar caso donde no retorna tupla (compatibilidad)
            if isinstance(result, tuple) and len(result) == 2:
                exec_df, new_executed_trades = result
            else:
                exec_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                new_executed_trades = []
            
            # REQUISITO 1: Actualizar lista de trades ejecutados
            if new_executed_trades:
                executed_trades.extend(new_executed_trades)
        except Exception as e:
            logger.error(f"  Error en simulación de latencia: {e}")
            exec_df = pd.DataFrame()
            new_executed_trades = []
        
        # Mostrar resumen de ejecuciones
        if exec_df is not None and len(exec_df) > 0:
            profitable = exec_df[exec_df['profit_category'] == 'Profitable'].copy()
            if len(profitable) > 0:
                print(f"\n  [RESUMEN DE EJECUCIONES PROFITABLES]")
                exec_summary_cols = ['epoch', 'execution_epoch', 'venue_max_bid', 'venue_min_ask',
                                    'executed_qty', 'real_profit', 'real_total_profit', 'profit_category']
                available_exec_cols = [c for c in exec_summary_cols if c in profitable.columns]
                display_dataframe(
                    profitable[available_exec_cols].head(10),
                    f"Primeras 10 Ejecuciones Profitable (de {len(profitable)} totales)",
                    max_rows=10
                )
        
        print("\n  [ANÁLISIS DE SENSIBILIDAD A LATENCIA]")
        sensitivity_df = sim.sensitivity_analysis(
            signals_df, 
            consolidated_tape,
            latencies_us=[10, 50, 100, 200, 500, 1000]
        )
        if sensitivity_df is not None and len(sensitivity_df) > 0:
            display_dataframe(sensitivity_df, "Sensibilidad a Diferentes Latencias", max_rows=10)
        
            # NO mostrar visualización individual - se harán gráficas agregadas al final
        
        print("\n  [EXPORTANDO RESULTADOS DE EJECUCIÓN]")
        sim.export_execution_results(
            exec_df,
            output_path=config.OUTPUT_DIR / f"execution_{isin}.csv"
        )
        
        # ====================================================================
        # FASE 6: ANÁLISIS FINAL
        # ====================================================================
        print(f"\n[FASE 6] Análisis final para {isin}...")
        analyzer = ArbitrageAnalyzer()
        
        print("\n  [ANALIZANDO OPORTUNIDADES]")
        metrics = analyzer.analyze_opportunities(signals_df, exec_df)
        
        print("\n  [ESTIMANDO ROI]")
        roi_metrics = analyzer.estimate_roi(metrics, trading_costs_bps=0.5, capital_eur=100000)
        
        # Mostrar métricas
        if metrics:
            print("\n  [MÉTRICAS DEL ANÁLISIS]")
            metrics_df = pd.DataFrame([metrics])
            display_dataframe(metrics_df, "Métricas Agregadas", max_rows=10)
        
        if roi_metrics:
            print("\n  [MÉTRICAS DE ROI]")
            roi_df = pd.DataFrame([roi_metrics])
            display_dataframe(roi_df, "Métricas de ROI", max_rows=10)
        
        # NO generar reporte individual .txt - se generará un único .md al final
        # con todos los ISINs que tengan oportunidades
        
        # ====================================================================
        # RESULTADO
        # ====================================================================
        results['success'] = True
        results['metrics'] = {
            'total_snapshots': len(consolidated_tape),
            'venues': len(clean_data),
            'opportunities': signals_df['is_rising_edge'].sum() if signals_df is not None else 0,
            'profitable_ops': (exec_df['profit_category'] == 'Profitable').sum() if exec_df is not None else 0,
            'theoretical_profit': signals_df[signals_df['is_rising_edge']]['total_profit'].sum() if signals_df is not None else 0,
            'real_profit': exec_df['real_total_profit'].sum() if exec_df is not None else 0,
            'roi_pct': roi_metrics['roi_pct'] if roi_metrics else 0
        }
        
        # Guardar datos para gráficas agregadas (solo referencias, no copias completas para ahorrar memoria)
        # Nota: Para datasets grandes, solo guardamos métricas, no los DataFrames completos
        results['has_data'] = True
        
        print(f"\n[ÉXITO] {isin} procesado correctamente")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"[ERROR] Error procesando {isin}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return results


def generate_aggregated_visualizations(all_results: List[Dict], max_plots: int = 10):
    """
    Genera gráficas agregadas de todos los ISINs procesados.
    Máximo 10 gráficas como se solicita.
    
    Args:
        all_results: Lista de resultados de todos los ISINs
        max_plots: Número máximo de gráficas a generar (default: 10)
    """
    print("\n" + "=" * 80)
    print("GENERANDO GRÁFICAS AGREGADAS")
    print("=" * 80)
    
    # Filtrar solo ISINs exitosos con métricas válidas
    successful = []
    for r in all_results:
        if r['success'] and r.get('metrics') is not None:
            successful.append(r)
    
    if len(successful) == 0:
        print("  [SKIP] No hay datos suficientes para generar gráficas agregadas")
        return
    
    plot_count = 0
    
    # ========================================================================
    # GRÁFICA 1: Profit Total por ISIN
    # ========================================================================
    if plot_count < max_plots:
        print(f"\n  [Gráfica {plot_count + 1}/{max_plots}] Profit Total por ISIN")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        isins = [r['isin'] for r in successful]
        theoretical_profits = [r['metrics'].get('theoretical_profit', 0) for r in successful]
        real_profits = [r['metrics'].get('real_profit', 0) for r in successful]
        
        x = np.arange(len(isins))
        width = 0.35
        
        ax.bar(x - width/2, theoretical_profits, width, label='Profit Teórico', alpha=0.7, color='green')
        ax.bar(x + width/2, real_profits, width, label='Profit Real', alpha=0.7, color='blue')
        
        ax.set_xlabel('ISIN')
        ax.set_ylabel('Profit (€)')
        ax.set_title('Profit Total por ISIN (Agregado)')
        ax.set_xticks(x)
        ax.set_xticklabels(isins, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = config.FIGURES_DIR / 'aggregated_profit_by_isin.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Guardada: {output_path}")
        plt.close()
        plot_count += 1
    
    # ========================================================================
    # GRÁFICA 2: Oportunidades Detectadas vs Profitable
    # ========================================================================
    if plot_count < max_plots:
        print(f"\n  [Gráfica {plot_count + 1}/{max_plots}] Oportunidades vs Profitable")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        isins = [r['isin'] for r in successful]
        total_opps = [r['metrics'].get('opportunities', 0) for r in successful]
        profitable_opps = [r['metrics'].get('profitable_ops', 0) for r in successful]
        
        x = np.arange(len(isins))
        width = 0.35
        
        ax.bar(x - width/2, total_opps, width, label='Total Oportunidades', alpha=0.7, color='orange')
        ax.bar(x + width/2, profitable_opps, width, label='Profitable', alpha=0.7, color='green')
        
        ax.set_xlabel('ISIN')
        ax.set_ylabel('Número de Oportunidades')
        ax.set_title('Oportunidades Detectadas vs Profitable por ISIN')
        ax.set_xticks(x)
        ax.set_xticklabels(isins, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = config.FIGURES_DIR / 'aggregated_opportunities.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Guardada: {output_path}")
        plt.close()
        plot_count += 1
    
    # ========================================================================
    # GRÁFICA 3: ROI por ISIN
    # ========================================================================
    if plot_count < max_plots:
        print(f"\n  [Gráfica {plot_count + 1}/{max_plots}] ROI por ISIN")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        isins = [r['isin'] for r in successful]
        rois = [r['metrics'].get('roi_pct', 0) for r in successful]
        
        colors = ['green' if roi > 0 else 'red' for roi in rois]
        ax.bar(isins, rois, alpha=0.7, color=colors)
        
        ax.set_xlabel('ISIN')
        ax.set_ylabel('ROI (%)')
        ax.set_title('ROI por ISIN')
        ax.set_xticklabels(isins, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = config.FIGURES_DIR / 'aggregated_roi_by_isin.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Guardada: {output_path}")
        plt.close()
        plot_count += 1
    
    # ========================================================================
    # GRÁFICA 4: Distribución de Profits (Histograma Agregado)
    # ========================================================================
    if plot_count < max_plots:
        print(f"\n  [Gráfica {plot_count + 1}/{max_plots}] Distribución de Profits")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Profit teórico (usar métricas agregadas)
        all_theoretical = []
        for r in successful:
            if r.get('metrics') is not None:
                # Usar métricas en lugar de DataFrame completo
                theoretical_profit = r['metrics'].get('theoretical_profit', 0)
                opportunities = r['metrics'].get('opportunities', 0)
                if opportunities > 0 and theoretical_profit > 0:
                    # Aproximar distribución con promedio
                    avg_profit = theoretical_profit / opportunities
                    all_theoretical.extend([avg_profit] * min(opportunities, 100))  # Limitar para no usar demasiada memoria
        
        if len(all_theoretical) > 0:
            axes[0].hist(all_theoretical, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[0].set_xlabel('Profit Teórico (€)')
            axes[0].set_ylabel('Frecuencia')
            axes[0].set_title('Distribución de Profit Teórico (Todos los ISINs)')
            axes[0].grid(True, alpha=0.3)
        
        # Profit real (usar métricas agregadas)
        all_real = []
        for r in successful:
            if r.get('metrics') is not None:
                # Usar métricas en lugar de DataFrame completo
                real_profit = r['metrics'].get('real_profit', 0)
                profitable_ops = r['metrics'].get('profitable_ops', 0)
                if profitable_ops > 0 and real_profit > 0:
                    # Aproximar distribución con promedio
                    avg_profit = real_profit / profitable_ops
                    all_real.extend([avg_profit] * min(profitable_ops, 100))  # Limitar para no usar demasiada memoria
        
        if len(all_real) > 0:
            axes[1].hist(all_real, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1].set_xlabel('Profit Real (€)')
            axes[1].set_ylabel('Frecuencia')
            axes[1].set_title('Distribución de Profit Real (Todos los ISINs)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = config.FIGURES_DIR / 'aggregated_profit_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Guardada: {output_path}")
        plt.close()
        plot_count += 1
    
    # ========================================================================
    # GRÁFICA 5: Tasa de Conversión (Oportunidades -> Profitable)
    # ========================================================================
    if plot_count < max_plots:
        print(f"\n  [Gráfica {plot_count + 1}/{max_plots}] Tasa de Conversión")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        isins = [r['isin'] for r in successful]
        conversion_rates = []
        for r in successful:
            if r.get('metrics') is not None:
                total = r['metrics'].get('opportunities', 0)
                profitable = r['metrics'].get('profitable_ops', 0)
                rate = (profitable / total * 100) if total > 0 else 0
                conversion_rates.append(rate)
            else:
                conversion_rates.append(0)
        
        ax.bar(isins, conversion_rates, alpha=0.7, color='purple')
        ax.set_xlabel('ISIN')
        ax.set_ylabel('Tasa de Conversión (%)')
        ax.set_title('Tasa de Conversión: Oportunidades -> Profitable')
        ax.set_xticklabels(isins, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = config.FIGURES_DIR / 'aggregated_conversion_rate.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Guardada: {output_path}")
        plt.close()
        plot_count += 1
    
    # ========================================================================
    # GRÁFICA 6: Profit Acumulado en el Tiempo (si hay datos de tiempo)
    # ========================================================================
    if plot_count < max_plots:
        print(f"\n  [Gráfica {plot_count + 1}/{max_plots}] Profit Acumulado")
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Agregar profits acumulados de todos los ISINs
        cumulative_profit = 0
        cumulative_data = []
        
        for r in successful:
            if r.get('metrics') is not None:
                # Usar métricas en lugar de DataFrame completo
                real_profit = r['metrics'].get('real_profit', 0)
                cumulative_profit += real_profit
                cumulative_data.append(cumulative_profit)
        
        if len(cumulative_data) > 0:
            ax.plot(range(len(cumulative_data)), cumulative_data, marker='o', linewidth=2, markersize=6)
            ax.fill_between(range(len(cumulative_data)), cumulative_data, alpha=0.3)
            ax.set_xlabel('ISINs Procesados (orden)')
            ax.set_ylabel('Profit Acumulado (€)')
            ax.set_title('Profit Acumulado por ISIN Procesado')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = config.FIGURES_DIR / 'aggregated_cumulative_profit.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Guardada: {output_path}")
        plt.close()
        plot_count += 1
    
    print(f"\n  [OK] Generadas {plot_count} gráficas agregadas (máximo {max_plots})")


def generate_comprehensive_summary_document(all_results: List[Dict], output_path: Path):
    """
    Genera un documento Markdown completo de resumen con todos los pasos, tablas y gráficas.
    
    Args:
        all_results: Lista de resultados de todos los ISINs
        output_path: Path para guardar el documento (.md)
    """
    print("\n" + "=" * 80)
    print("GENERANDO DOCUMENTO MARKDOWN ÚNICO COMPLETO")
    print("=" * 80)
    
    # Filtrar solo ISINs exitosos que tengan oportunidades
    successful_with_opps = []
    for r in all_results:
        if r['success'] and r.get('metrics'):
            opportunities = r['metrics'].get('opportunities', 0)
            if opportunities > 0:
                successful_with_opps.append(r)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"  ISINs con oportunidades: {len(successful_with_opps)}")
    print(f"  ISINs exitosos sin oportunidades: {len(successful) - len(successful_with_opps)}")
    print(f"  ISINs fallidos: {len(failed)}")
    
    doc_lines = []
    
    # ========================================================================
    # HEADER (Markdown)
    # ========================================================================
    doc_lines.append("# DOCUMENTO DE RESUMEN COMPLETO - ANÁLISIS DE ARBITRAJE HFT")
    doc_lines.append("")
    doc_lines.append(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc_lines.append(f"**Total ISINs procesados:** {len(all_results)}")
    doc_lines.append(f"**ISINs con oportunidades:** {len(successful_with_opps)}")
    doc_lines.append(f"**ISINs exitosos sin oportunidades:** {len(successful) - len(successful_with_opps)}")
    doc_lines.append(f"**ISINs fallidos:** {len(failed)}")
    doc_lines.append("")
    doc_lines.append("> **Nota:** Este documento incluye únicamente ISINs que tienen oportunidades detectadas.")
    doc_lines.append("")
    doc_lines.append("---")
    doc_lines.append("")
    
    # ========================================================================
    # RESUMEN EJECUTIVO
    # ========================================================================
    doc_lines.append("## RESUMEN EJECUTIVO")
    doc_lines.append("")
    
    if len(successful_with_opps) > 0:
        total_snapshots = sum(r['metrics']['total_snapshots'] for r in successful_with_opps)
        total_opportunities = sum(r['metrics']['opportunities'] for r in successful_with_opps)
        total_profitable = sum(r['metrics']['profitable_ops'] for r in successful_with_opps)
        total_theoretical = sum(r['metrics']['theoretical_profit'] for r in successful_with_opps)
        total_real = sum(r['metrics']['real_profit'] for r in successful_with_opps)
        avg_roi = sum(r['metrics']['roi_pct'] for r in successful_with_opps) / len(successful_with_opps) if len(successful_with_opps) > 0 else 0
        
        doc_lines.append(f"- **Total snapshots analizados:** {total_snapshots:,}")
        doc_lines.append(f"- **Total oportunidades detectadas:** {total_opportunities:,}")
        doc_lines.append(f"- **Total oportunidades profitable:** {total_profitable:,}")
        doc_lines.append(f"- **Tasa de conversión:** {(total_profitable/total_opportunities*100):.2f}%" if total_opportunities > 0 else "N/A")
        doc_lines.append(f"- **Profit teórico total:** €{total_theoretical:,.2f}")
        doc_lines.append(f"- **Profit real total:** €{total_real:,.2f}")
        doc_lines.append(f"- **ROI promedio:** {avg_roi:.4f}%")
        doc_lines.append("")
    
    # ========================================================================
    # TABLA COMPLETA DE TODAS LAS OPORTUNIDADES
    # ========================================================================
    doc_lines.append("## TABLA COMPLETA DE TODAS LAS OPORTUNIDADES")
    doc_lines.append("")
    
    all_opportunities = []
    for r in successful_with_opps:
        isin = r['isin']
        opp_file = config.OUTPUT_DIR / f"opportunities_{isin}.csv"
        if opp_file.exists():
            try:
                opp_df = pd.read_csv(opp_file)
                for _, opp in opp_df.iterrows():
                    all_opportunities.append({
                        'ISIN': isin,
                        'Epoch': opp.get('epoch', 'N/A'),
                        'Venue_Buy': opp.get('venue_min_ask', 'N/A'),
                        'Venue_Sell': opp.get('venue_max_bid', 'N/A'),
                        'Executable_Qty': opp.get('executable_qty', 0),
                        'Theoretical_Profit': opp.get('theoretical_profit', 0),
                        'Total_Profit': opp.get('total_profit', 0),
                        'Bid_Qty': opp.get('bid_qty', 0),
                        'Ask_Qty': opp.get('ask_qty', 0)
                    })
            except Exception as e:
                logger.warning(f"  No se pudo cargar oportunidades de {isin}: {e}")
    
    if len(all_opportunities) > 0:
        opp_df_all = pd.DataFrame(all_opportunities)
        doc_lines.append(f"**Total oportunidades detectadas:** {len(all_opportunities):,}")
        doc_lines.append("")
        doc_lines.append("### Todas las Oportunidades")
        doc_lines.append("")
        if HAS_MARKDOWN:
            doc_lines.append(opp_df_all.to_markdown(index=False))
        else:
            doc_lines.append("```")
            doc_lines.append(opp_df_all.to_string(index=False))
            doc_lines.append("```")
        doc_lines.append("")
    
    # ========================================================================
    # TABLA DE TRADES ACUMULADOS
    # ========================================================================
    doc_lines.append("## TABLA DE TRADES ACUMULADOS")
    doc_lines.append("")
    
    # Cargar trades desde archivos CSV generados
    all_trades = []
    for r in successful_with_opps:
        isin = r['isin']
        exec_file = config.OUTPUT_DIR / f"execution_{isin}.csv"
        if exec_file.exists():
            try:
                exec_df = pd.read_csv(exec_file)
                profitable = exec_df[exec_df['profit_category'] == 'Profitable'].copy()
                if len(profitable) > 0:
                    for _, trade in profitable.iterrows():
                        all_trades.append({
                            'ISIN': isin,
                            'Epoch': trade.get('epoch', 'N/A'),
                            'Execution_Epoch': trade.get('execution_epoch', 'N/A'),
                            'Venue_Buy': trade.get('venue_min_ask', 'N/A'),
                            'Venue_Sell': trade.get('venue_max_bid', 'N/A'),
                            'Executed_Qty': trade.get('executed_qty', 0),
                            'Real_Profit_Per_Unit': trade.get('real_profit', 0),
                            'Real_Total_Profit': trade.get('real_total_profit', 0),
                            'Profit_Category': trade.get('profit_category', 'N/A')
                        })
            except Exception as e:
                logger.warning(f"  No se pudo cargar trades de {isin}: {e}")
    
    if len(all_trades) > 0:
        trades_df = pd.DataFrame(all_trades)
        
        # Resumen de trades
        doc_lines.append(f"Total trades ejecutados: {len(all_trades):,}")
        doc_lines.append(f"Total profit acumulado: €{trades_df['Real_Total_Profit'].sum():,.2f}")
        doc_lines.append(f"Profit promedio por trade: €{trades_df['Real_Total_Profit'].mean():,.2f}")
        doc_lines.append(f"Profit máximo: €{trades_df['Real_Total_Profit'].max():,.2f}")
        doc_lines.append(f"Cantidad total ejecutada: {trades_df['Executed_Qty'].sum():,}")
        doc_lines.append("")
        
        # Top 20 trades
        doc_lines.append("### Top 20 Trades por Profit")
        doc_lines.append("")
        top_trades = trades_df.nlargest(20, 'Real_Total_Profit')
        if HAS_MARKDOWN:
            doc_lines.append(top_trades.to_markdown(index=False))
        else:
            doc_lines.append("```")
            doc_lines.append(top_trades.to_string(index=False))
            doc_lines.append("```")
        doc_lines.append("")
        
        # Estadísticas por ISIN
        doc_lines.append("### Estadísticas de Trades por ISIN")
        doc_lines.append("")
        isin_stats = trades_df.groupby('ISIN').agg({
            'Real_Total_Profit': ['count', 'sum', 'mean', 'max'],
            'Executed_Qty': 'sum'
        }).round(2)
        isin_stats.columns = ['Num_Trades', 'Total_Profit', 'Avg_Profit', 'Max_Profit', 'Total_Qty']
        if HAS_MARKDOWN:
            doc_lines.append(isin_stats.to_markdown())
        else:
            doc_lines.append("```")
            doc_lines.append(isin_stats.to_string())
            doc_lines.append("```")
        doc_lines.append("")
    else:
        doc_lines.append("No hay trades ejecutados para mostrar.")
        doc_lines.append("")
    
    # ========================================================================
    # TABLA DE RESULTADOS ACUMULADOS POR ISIN (Top 5)
    # ========================================================================
    doc_lines.append("## TABLA DE RESULTADOS ACUMULADOS POR ISIN")
    doc_lines.append("")
    doc_lines.append("### Top 5 ISINs por Profit Real")
    doc_lines.append("")
    
    # ========================================================================
    # DETALLE POR ISIN (solo los que tienen oportunidades)
    # ========================================================================
    doc_lines.append("## DETALLE POR ISIN")
    doc_lines.append("")
    doc_lines.append("A continuación se muestran los detalles de cada ISIN que tiene oportunidades detectadas:")
    doc_lines.append("")
    
    results_data = []
    for r in successful_with_opps:
        m = r['metrics']
        results_data.append({
            'ISIN': r['isin'],
            'Snapshots': m['total_snapshots'],
            'Venues': m['venues'],
            'Opportunities': m['opportunities'],
            'Profitable_Ops': m['profitable_ops'],
            'Conversion_Rate_%': (m['profitable_ops'] / m['opportunities'] * 100) if m['opportunities'] > 0 else 0,
            'Theoretical_Profit_EUR': m['theoretical_profit'],
            'Real_Profit_EUR': m['real_profit'],
            'Profit_Loss_EUR': m['theoretical_profit'] - m['real_profit'],
            'ROI_%': m['roi_pct']
        })
    
    if len(results_data) > 0:
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Real_Profit_EUR', ascending=False)
        
        # Mostrar tabla resumen (top 5 en tabla compacta)
        doc_lines.append("### Tabla Resumen (Top 5)")
        doc_lines.append("")
        top_5_df = results_df.head(5)
        if HAS_MARKDOWN:
            doc_lines.append(top_5_df.to_markdown(index=False))
        else:
            doc_lines.append("```")
            doc_lines.append(top_5_df.to_string(index=False))
            doc_lines.append("```")
        doc_lines.append("")
        
        # Detalle completo de cada ISIN
        doc_lines.append("### Detalle Completo por ISIN")
        doc_lines.append("")
        
        for idx, r in enumerate(successful_with_opps, 1):
            isin = r['isin']
            m = r['metrics']
            
            doc_lines.append(f"#### {idx}. ISIN: {isin}")
            doc_lines.append("")
            doc_lines.append(f"- **Snapshots:** {m['total_snapshots']:,}")
            doc_lines.append(f"- **Venues:** {m['venues']}")
            doc_lines.append(f"- **Oportunidades detectadas:** {m['opportunities']:,}")
            doc_lines.append(f"- **Oportunidades profitable:** {m['profitable_ops']:,}")
            doc_lines.append(f"- **Tasa de conversión:** {(m['profitable_ops']/m['opportunities']*100):.2f}%" if m['opportunities'] > 0 else "N/A")
            doc_lines.append(f"- **Profit teórico:** €{m['theoretical_profit']:,.2f}")
            doc_lines.append(f"- **Profit real:** €{m['real_profit']:,.2f}")
            doc_lines.append(f"- **ROI:** {m['roi_pct']:.4f}%")
            doc_lines.append("")
            
            # Cargar y mostrar oportunidades de este ISIN
            opp_file = config.OUTPUT_DIR / f"opportunities_{isin}.csv"
            if opp_file.exists():
                try:
                    opp_df = pd.read_csv(opp_file)
                    if len(opp_df) > 0:
                        doc_lines.append(f"**Oportunidades de {isin}:**")
                        doc_lines.append("")
                        opp_cols = ['epoch', 'venue_max_bid', 'venue_min_ask', 'executable_qty', 
                                   'theoretical_profit', 'total_profit']
                        available_opp_cols = [c for c in opp_cols if c in opp_df.columns]
                        if HAS_MARKDOWN:
                            doc_lines.append(opp_df[available_opp_cols].to_markdown(index=False))
                        else:
                            doc_lines.append("```")
                            doc_lines.append(opp_df[available_opp_cols].to_string(index=False))
                            doc_lines.append("```")
                        doc_lines.append("")
                except Exception as e:
                    logger.warning(f"  No se pudo cargar oportunidades de {isin}: {e}")
            
            # Cargar y mostrar trades de este ISIN
            exec_file = config.OUTPUT_DIR / f"execution_{isin}.csv"
            if exec_file.exists():
                try:
                    exec_df = pd.read_csv(exec_file)
                    profitable_trades = exec_df[exec_df['profit_category'] == 'Profitable'].copy()
                    if len(profitable_trades) > 0:
                        doc_lines.append(f"**Trades ejecutados de {isin}:**")
                        doc_lines.append("")
                        trade_cols = ['epoch', 'execution_epoch', 'venue_max_bid', 'venue_min_ask',
                                     'executed_qty', 'real_profit', 'real_total_profit']
                        available_trade_cols = [c for c in trade_cols if c in profitable_trades.columns]
                        if HAS_MARKDOWN:
                            doc_lines.append(profitable_trades[available_trade_cols].to_markdown(index=False))
                        else:
                            doc_lines.append("```")
                            doc_lines.append(profitable_trades[available_trade_cols].to_string(index=False))
                            doc_lines.append("```")
                        doc_lines.append("")
                except Exception as e:
                    logger.warning(f"  No se pudo cargar trades de {isin}: {e}")
            
            doc_lines.append("---")
            doc_lines.append("")
        
        # Totales
        doc_lines.append("### Totales Acumulados")
        doc_lines.append("")
        doc_lines.append(f"- **Total Snapshots:** {results_df['Snapshots'].sum():,}")
        doc_lines.append(f"- **Total Opportunities:** {results_df['Opportunities'].sum():,}")
        doc_lines.append(f"- **Total Profitable Ops:** {results_df['Profitable_Ops'].sum():,}")
        doc_lines.append(f"- **Total Theoretical Profit:** €{results_df['Theoretical_Profit_EUR'].sum():,.2f}")
        doc_lines.append(f"- **Total Real Profit:** €{results_df['Real_Profit_EUR'].sum():,.2f}")
        doc_lines.append(f"- **Total Profit Loss:** €{results_df['Profit_Loss_EUR'].sum():,.2f}")
        doc_lines.append(f"- **Average ROI:** {results_df['ROI_%'].mean():.4f}%")
        doc_lines.append("")
    
    # ========================================================================
    # GRÁFICAS GENERADAS
    # ========================================================================
    doc_lines.append("## GRÁFICAS GENERADAS")
    doc_lines.append("")
    doc_lines.append("Las siguientes gráficas agregadas han sido generadas:")
    doc_lines.append("")
    
    figure_files = list(config.FIGURES_DIR.glob('aggregated_*.png'))
    for fig_file in sorted(figure_files):
        fig_name = fig_file.stem.replace('aggregated_', '').replace('_', ' ').title()
        doc_lines.append(f"### {fig_name}")
        doc_lines.append("")
        doc_lines.append(f"![{fig_name}]({fig_file.relative_to(config.OUTPUT_DIR.parent)})")
        doc_lines.append("")
    
    # ========================================================================
    # PASOS DEL PROCESAMIENTO
    # ========================================================================
    doc_lines.append("## PASOS DEL PROCESAMIENTO")
    doc_lines.append("")
    doc_lines.append("### Fase 1: Carga de Datos")
    doc_lines.append("- Carga de archivos QTE (quotes) y STS (status) por venue")
    doc_lines.append("- Validación de columnas requeridas")
    doc_lines.append("- Conversión de tipos de datos")
    doc_lines.append("")
    
    doc_lines.append("### Fase 2: Limpieza y Validación")
    doc_lines.append("- Filtrado de magic numbers")
    doc_lines.append("- Validación de precios y cantidades")
    doc_lines.append("- Detección de crossed books")
    doc_lines.append("- Filtrado por market status (solo continuous trading)")
    doc_lines.append("- Generación de columna seq para eventos con mismo epoch")
    doc_lines.append("")
    
    doc_lines.append("### Fase 3: Consolidated Tape")
    doc_lines.append("- Merge de datos multi-venue usando merge_asof (direction='backward')")
    doc_lines.append("- Redondeo temporal a bins")
    doc_lines.append("- Forward fill para propagar últimos precios conocidos")
    doc_lines.append("- Validación del tape consolidado")
    doc_lines.append("")
    
    doc_lines.append("### Fase 4: Detección de Señales")
    doc_lines.append("- Cálculo de Global Max Bid y Global Min Ask")
    doc_lines.append("- Detección de condición de arbitraje (Bid > Ask)")
    doc_lines.append("- Cálculo de profit teórico y cantidad ejecutable")
    doc_lines.append("- Rising Edge Detection (evitar doble conteo)")
    doc_lines.append("- Filtrado de oportunidades ya ejecutadas")
    doc_lines.append("- Soporte de ejecución parcial")
    doc_lines.append("")
    
    doc_lines.append("### Fase 5: Simulación de Latencia")
    doc_lines.append("- Time Machine: buscar precios en T + latency")
    doc_lines.append("- Re-cálculo de profit con precios time-shifted")
    doc_lines.append("- Clasificación: Profitable / Break-even / Loss")
    doc_lines.append("- Registro de trades ejecutados")
    doc_lines.append("- Análisis de sensibilidad a diferentes latencias")
    doc_lines.append("")
    
    doc_lines.append("### Fase 6: Análisis Final")
    doc_lines.append("- Análisis de oportunidades")
    doc_lines.append("- Estimación de ROI")
    doc_lines.append("- Generación de reportes")
    doc_lines.append("")
    
    # ========================================================================
    # ISINs FALLIDOS
    # ========================================================================
    if len(failed) > 0:
        doc_lines.append("## ISINs FALLIDOS")
        doc_lines.append("")
        
        for r in failed:
            doc_lines.append(f"### {r['isin']}")
            doc_lines.append(f"**Error:** {r['error']}")
            doc_lines.append("")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    doc_lines.append("---")
    doc_lines.append("")
    doc_lines.append("*Documento generado automáticamente por el sistema de análisis de arbitraje HFT*")
    
    # Guardar documento
    doc_text = "\n".join(doc_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc_text)
    
    print(f"\n  [OK] Documento de resumen completo guardado en: {output_path}")
    print(f"    - Total trades documentados: {len(all_trades):,}")
    print(f"    - Total ISINs documentados: {len(successful):,}")


def generate_consolidated_report(all_results: list, output_path: Path):
    """
    Genera un reporte consolidado de todos los ISINs procesados.
    
    Args:
        all_results: Lista de dicts con resultados por ISIN
        output_path: Path para guardar el reporte
    """
    
    print("\n" + "=" * 80)
    print("GENERANDO REPORTE CONSOLIDADO")
    print("=" * 80)
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("REPORTE CONSOLIDADO - ANÁLISIS MULTI-ISIN (DATA_BIG)")
    report_lines.append("=" * 80)
    report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total ISINs procesados: {len(all_results)}")
    report_lines.append("")
    
    # Separar exitosos de fallidos
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    report_lines.append(f"Exitosos: {len(successful)}")
    report_lines.append(f"Fallidos: {len(failed)}")
    report_lines.append("")
    
    # ========================================================================
    # RESUMEN AGREGADO
    # ========================================================================
    if len(successful) > 0:
        report_lines.append("=" * 80)
        report_lines.append("RESUMEN AGREGADO (TODOS LOS ISINs)")
        report_lines.append("=" * 80)
        
        total_snapshots = sum(r['metrics']['total_snapshots'] for r in successful)
        total_opportunities = sum(r['metrics']['opportunities'] for r in successful)
        total_profitable = sum(r['metrics']['profitable_ops'] for r in successful)
        total_theoretical = sum(r['metrics']['theoretical_profit'] for r in successful)
        total_real = sum(r['metrics']['real_profit'] for r in successful)
        avg_roi = sum(r['metrics']['roi_pct'] for r in successful) / len(successful)
        
        report_lines.append(f"Total snapshots analizados: {total_snapshots:,}")
        report_lines.append(f"Total oportunidades detectadas: {total_opportunities:,}")
        report_lines.append(f"Total oportunidades profitable: {total_profitable:,}")
        report_lines.append(f"Profit teórico total: €{total_theoretical:.2f}")
        report_lines.append(f"Profit real total: €{total_real:.2f}")
        report_lines.append(f"ROI promedio: {avg_roi:.4f}%")
        report_lines.append("")
    
    # ========================================================================
    # DETALLE POR ISIN
    # ========================================================================
    if len(successful) > 0:
        report_lines.append("=" * 80)
        report_lines.append("DETALLE POR ISIN")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Ordenar por profit real (descendente)
        successful_sorted = sorted(
            successful, 
            key=lambda x: x['metrics']['real_profit'], 
            reverse=True
        )
        
        for r in successful_sorted:
            m = r['metrics']
            report_lines.append(f"ISIN: {r['isin']}")
            report_lines.append(f"  Snapshots: {m['total_snapshots']:,}")
            report_lines.append(f"  Venues: {m['venues']}")
            report_lines.append(f"  Oportunidades: {m['opportunities']:,}")
            report_lines.append(f"  Profitable: {m['profitable_ops']:,}")
            report_lines.append(f"  Profit teórico: €{m['theoretical_profit']:.2f}")
            report_lines.append(f"  Profit real: €{m['real_profit']:.2f}")
            report_lines.append(f"  ROI: {m['roi_pct']:.4f}%")
            report_lines.append("")
    
    # ========================================================================
    # ISINS FALLIDOS
    # ========================================================================
    if len(failed) > 0:
        report_lines.append("=" * 80)
        report_lines.append("ISINs FALLIDOS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for r in failed:
            report_lines.append(f"ISIN: {r['isin']}")
            report_lines.append(f"  Error: {r['error']}")
            report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("[FIN DEL REPORTE CONSOLIDADO]")
    report_lines.append("=" * 80)
    
    # Guardar
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n[ÉXITO] Reporte consolidado guardado en: {output_path}")
    print("\nRESUMEN:")
    print(f"  Total ISINs: {len(all_results)}")
    print(f"  Exitosos: {len(successful)}")
    print(f"  Fallidos: {len(failed)}")
    
    if len(successful) > 0:
        total_profit = sum(r['metrics']['real_profit'] for r in successful)
        print(f"  Profit real total: €{total_profit:.2f}")


def main():
    """
    Función principal para procesamiento de DATA_BIG.
    """
    
    # ========================================================================
    # ARGUMENTOS DE LÍNEA DE COMANDOS
    # ========================================================================
    parser = argparse.ArgumentParser(description='Procesar DATA_BIG para arbitraje HFT')
    parser.add_argument('--max-isins', type=int, default=None,
                       help='Número máximo de ISINs a procesar (None = todos)')
    parser.add_argument('--time-bin', type=int, default=200,
                       help='Time bin en milisegundos para consolidación')
    parser.add_argument('--specific-isin', type=str, default=None,
                       help='Procesar solo un ISIN específico')
    
    args = parser.parse_args()
    
    # ========================================================================
    # INICIALIZACIÓN
    # ========================================================================
    print("=" * 80)
    print("SISTEMA DE DETECCIÓN DE ARBITRAJE HFT - PRODUCCIÓN (DATA_BIG)")
    print("=" * 80)
    
    logger.info("Inicializando sistema para DATA_BIG...")
    
    # Usar DATA_BIG
    data_dir = config.DATA_BIG_DIR
    
    logger.info(f"Directorio de datos: {data_dir}")
    print(f"Time bin: {args.time_bin} ms")
    
    # Verificar que el directorio existe
    if not data_dir.exists():
        logger.error(f"[ERROR] Directorio no encontrado: {data_dir}")
        logger.error("Por favor, asegúrate de que DATA_BIG está en la ubicación correcta")
        return
    
    # Limpiar outputs anteriores
    clean_output_directories()
    
    # Crear directorios de output (ya están limpios)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    # ========================================================================
    # DESCUBRIMIENTO DE ISINs
    # ========================================================================
    loader = DataLoader(data_dir)
    
    if args.specific_isin:
        # Procesar solo un ISIN específico
        isins_to_process = [args.specific_isin]
        print(f"\nProcesando ISIN específico: {args.specific_isin}")
    else:
        # Descubrir todos los ISINs
        all_isins = loader.discover_isins()
        
        if len(all_isins) == 0:
            logger.error("No se encontraron ISINs en DATA_BIG")
            return
        
        # Limitar número de ISINs si se especifica
        if args.max_isins:
            isins_to_process = all_isins[:args.max_isins]
            print(f"\nProcesando {len(isins_to_process)} ISINs de {len(all_isins)} disponibles")
        else:
            isins_to_process = all_isins
            print(f"\nProcesando TODOS los ISINs: {len(isins_to_process)}")
    
    print(f"ISINs a procesar: {isins_to_process}")
    
    # ========================================================================
    # PROCESAMIENTO MULTI-ISIN
    # ========================================================================
    all_results = []
    
    for idx, isin in enumerate(isins_to_process, 1):
        print("\n" + "=" * 80)
        print(f"PROGRESO: {idx}/{len(isins_to_process)} ISINs")
        print("=" * 80)
        
        result = process_single_isin(isin, loader, time_bin_ms=args.time_bin)
        all_results.append(result)
        
        # Liberar memoria
        import gc
        gc.collect()
    
    # ========================================================================
    # GRÁFICAS AGREGADAS
    # ========================================================================
    generate_aggregated_visualizations(all_results, max_plots=10)
    
    # ========================================================================
    # DOCUMENTO DE RESUMEN COMPLETO
    # ========================================================================
    comprehensive_summary_path = config.OUTPUT_DIR / 'comprehensive_summary_BIG.md'
    generate_comprehensive_summary_document(all_results, comprehensive_summary_path)
    
    # ========================================================================
    # REPORTE CONSOLIDADO (legacy)
    # ========================================================================
    consolidated_report_path = config.OUTPUT_DIR / 'consolidated_report_BIG.txt'
    generate_consolidated_report(all_results, consolidated_report_path)
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 80)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nEstadísticas finales:")
    print(f"  Total ISINs procesados: {len(all_results)}")
    print(f"  Exitosos: {len(successful)}")
    print(f"  Fallidos: {len(failed)}")
    
    if len(successful) > 0:
        total_profit = sum(r['metrics'].get('real_profit', 0) for r in successful if r.get('metrics') is not None)
        total_opportunities = sum(r['metrics'].get('opportunities', 0) for r in successful if r.get('metrics') is not None)
        
        print(f"\n  Total oportunidades detectadas: {total_opportunities:,}")
        print(f"  Profit real total: €{total_profit:.2f}")
    
    print(f"\nArchivos generados:")
    print(f"  - Log: {config.OUTPUT_DIR / 'arbitrage_system_BIG.log'}")
    print(f"  - Documento Markdown único: {comprehensive_summary_path}")
    print(f"  - Reporte consolidado (legacy): {consolidated_report_path}")
    print(f"  - Gráficas agregadas: {config.FIGURES_DIR / 'aggregated_*.png'}")
    print(f"  - Oportunidades CSV: {config.OUTPUT_DIR / 'opportunities_*.csv'}")
    print(f"  - Ejecuciones CSV: {config.OUTPUT_DIR / 'execution_*.csv'}")
    
    # Crear resumen CSV
    summary_data = []
    for r in successful:
        if r.get('metrics') is not None:
            summary_data.append({
                'ISIN': r['isin'],
                'Snapshots': r['metrics'].get('total_snapshots', 0),
                'Venues': r['metrics'].get('venues', 0),
                'Opportunities': r['metrics'].get('opportunities', 0),
                'Profitable_Ops': r['metrics'].get('profitable_ops', 0),
                'Theoretical_Profit_EUR': r['metrics'].get('theoretical_profit', 0),
                'Real_Profit_EUR': r['metrics'].get('real_profit', 0),
                'ROI_PCT': r['metrics'].get('roi_pct', 0)
            })
    
    if len(summary_data) > 0:
        summary_df = pd.DataFrame(summary_data)
        summary_path = config.OUTPUT_DIR / 'summary_BIG.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"  - Resumen CSV: {summary_path}")
    
    print("\n" + "=" * 80)
    print("[ÉXITO] PROCESAMIENTO DE DATA_BIG COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPCIÓN] Procesamiento cancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR FATAL] {e}", exc_info=True)
        sys.exit(1)
