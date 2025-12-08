"""
================================================================================
main.py - Script Principal del Sistema de Arbitraje HFT
================================================================================

Orquesta el pipeline completo:
1. Carga de datos (DataLoader)
2. Limpieza y validación (DataCleaner)
3. Consolidated Tape (ConsolidatedTape) - OPTIMIZADO
4. Detección de señales (SignalGenerator)
5. Simulación de latencia (próximo módulo)
6. Análisis y visualizaciones

Uso:
    python main.py --data-dir DATA_SMALL --isin <ISIN> --visualize

CORRECCIONES APLICADAS:
- Configuración UTF-8 para Windows
- Uso de consolidador optimizado (evita explosión de memoria)
- Limpieza adaptativa de datos
- Mejor manejo de errores

================================================================================
"""

import logging
import sys
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime

# Verificar si pandas tiene to_markdown
try:
    _test_df = pd.DataFrame({'a': [1]})
    _test_df.to_markdown()
    HAS_MARKDOWN = True
except:
    HAS_MARKDOWN = False

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
        logging.FileHandler(config.OUTPUT_DIR / 'arbitrage_system.log', encoding='utf-8'),
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
    
    # Cerrar todos los handlers de logging antes de eliminar archivos
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    cleaned_files = 0
    cleaned_dirs = 0
    
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
                print(f"  No se pudo eliminar {item}: {e}")
    
    # Asegurar que los directorios existen
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    # Reconfigurar logging después de limpiar
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.OUTPUT_DIR / 'arbitrage_system.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    print(f"  Archivos eliminados: {cleaned_files}")
    print(f"  [OK] Directorios de output listos para nueva ejecución")
    print("=" * 80)


def generate_complete_markdown_report(isin: str, consolidated_tape, signals_df, exec_df,
                                      venue_pairs, metrics, roi_metrics, clean_data):
    """
    Genera un documento Markdown completo con todos los pasos, tablas y gráficas.
    
    Args:
        isin: ISIN procesado
        consolidated_tape: DataFrame del tape consolidado
        signals_df: DataFrame de señales detectadas
        exec_df: DataFrame de ejecuciones
        venue_pairs: DataFrame de pares de venues
        metrics: Dict con métricas
        roi_metrics: Dict con métricas de ROI
        clean_data: Dict con datos limpios por venue
    """
    output_path = config.OUTPUT_DIR / f'complete_report_{isin}.md'
    
    doc_lines = []
    
    # Header
    doc_lines.append(f"# REPORTE COMPLETO - ISIN: {isin}")
    doc_lines.append("")
    doc_lines.append(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc_lines.append("")
    doc_lines.append("---")
    doc_lines.append("")
    
    # Resumen ejecutivo
    doc_lines.append("## RESUMEN EJECUTIVO")
    doc_lines.append("")
    doc_lines.append(f"- **ISIN:** {isin}")
    doc_lines.append(f"- **Total snapshots:** {len(consolidated_tape):,}")
    doc_lines.append(f"- **Venues incluidos:** {len(clean_data)}")
    
    if signals_df is not None and len(signals_df) > 0:
        total_opps = signals_df['is_rising_edge'].sum()
        total_profit = signals_df[signals_df['is_rising_edge']]['total_profit'].sum()
        doc_lines.append(f"- **Oportunidades detectadas:** {total_opps:,}")
        doc_lines.append(f"- **Profit teórico total:** €{total_profit:,.2f}")
    
    if exec_df is not None and len(exec_df) > 0:
        profitable = (exec_df['profit_category'] == 'Profitable').sum()
        real_profit = exec_df['real_total_profit'].sum()
        doc_lines.append(f"- **Oportunidades profitable:** {profitable:,}")
        doc_lines.append(f"- **Profit real total:** €{real_profit:,.2f}")
    
    if roi_metrics:
        doc_lines.append(f"- **ROI estimado:** {roi_metrics.get('roi_pct', 0):.4f}%")
    
    doc_lines.append("")
    doc_lines.append("---")
    doc_lines.append("")
    
    # Tabla completa de todas las oportunidades
    doc_lines.append("## TABLA COMPLETA DE TODAS LAS OPORTUNIDADES")
    doc_lines.append("")
    
    if signals_df is not None and len(signals_df) > 0:
        opps = signals_df[signals_df['is_rising_edge']].copy()
        if len(opps) > 0:
            opp_cols = ['epoch', 'venue_max_bid', 'venue_min_ask', 'executable_qty', 
                       'theoretical_profit', 'total_profit', 'bid_qty', 'ask_qty']
            available_cols = [c for c in opp_cols if c in opps.columns]
            if HAS_MARKDOWN:
                doc_lines.append(opps[available_cols].to_markdown(index=False))
            else:
                doc_lines.append("```")
                doc_lines.append(opps[available_cols].to_string(index=False))
                doc_lines.append("```")
            doc_lines.append("")
    
    # Tabla de trades ejecutados
    if exec_df is not None and len(exec_df) > 0:
        doc_lines.append("## TABLA DE TRADES EJECUTADOS")
        doc_lines.append("")
        profitable_trades = exec_df[exec_df['profit_category'] == 'Profitable'].copy()
        if len(profitable_trades) > 0:
            trade_cols = ['epoch', 'execution_epoch', 'venue_max_bid', 'venue_min_ask',
                         'executed_qty', 'real_profit', 'real_total_profit', 'profit_category']
            available_trade_cols = [c for c in trade_cols if c in profitable_trades.columns]
            if HAS_MARKDOWN:
                doc_lines.append(profitable_trades[available_trade_cols].to_markdown(index=False))
            else:
                doc_lines.append("```")
                doc_lines.append(profitable_trades[available_trade_cols].to_string(index=False))
                doc_lines.append("```")
            doc_lines.append("")
    
    # Pares de venues
    if venue_pairs is not None and len(venue_pairs) > 0:
        doc_lines.append("## ANÁLISIS DE PARES DE VENUES")
        doc_lines.append("")
        if HAS_MARKDOWN:
            doc_lines.append(venue_pairs.to_markdown(index=False))
        else:
            doc_lines.append("```")
            doc_lines.append(venue_pairs.to_string(index=False))
            doc_lines.append("```")
        doc_lines.append("")
    
    # Métricas
    if metrics:
        doc_lines.append("## MÉTRICAS DEL ANÁLISIS")
        doc_lines.append("")
        metrics_df = pd.DataFrame([metrics])
        if HAS_MARKDOWN:
            doc_lines.append(metrics_df.to_markdown(index=False))
        else:
            doc_lines.append("```")
            doc_lines.append(metrics_df.to_string(index=False))
            doc_lines.append("```")
        doc_lines.append("")
    
    if roi_metrics:
        doc_lines.append("## MÉTRICAS DE ROI")
        doc_lines.append("")
        roi_df = pd.DataFrame([roi_metrics])
        if HAS_MARKDOWN:
            doc_lines.append(roi_df.to_markdown(index=False))
        else:
            doc_lines.append("```")
            doc_lines.append(roi_df.to_string(index=False))
            doc_lines.append("```")
        doc_lines.append("")
    
    # Gráficas generadas
    doc_lines.append("## GRÁFICAS GENERADAS")
    doc_lines.append("")
    
    figure_files = [
        config.FIGURES_DIR / f'consolidated_tape_{isin}.png',
        config.FIGURES_DIR / f'signals_{isin}.png',
        config.FIGURES_DIR / f'latency_impact_{isin}.png'
    ]
    
    for fig_file in figure_files:
        if fig_file.exists():
            fig_name = fig_file.stem.replace(f'_{isin}', '').replace('_', ' ').title()
            doc_lines.append(f"### {fig_name}")
            doc_lines.append("")
            doc_lines.append(f"![{fig_name}]({fig_file.relative_to(config.OUTPUT_DIR.parent)})")
            doc_lines.append("")
    
    # Pasos del procesamiento
    doc_lines.append("## PASOS DEL PROCESAMIENTO")
    doc_lines.append("")
    doc_lines.append("### Fase 1: Carga de Datos")
    doc_lines.append("- Carga de archivos QTE y STS")
    doc_lines.append("- Validación de columnas")
    doc_lines.append("")
    doc_lines.append("### Fase 2: Limpieza")
    doc_lines.append("- Filtrado de magic numbers")
    doc_lines.append("- Validación de precios")
    doc_lines.append("- Filtrado por market status")
    doc_lines.append("")
    doc_lines.append("### Fase 3: Consolidated Tape")
    doc_lines.append("- Merge multi-venue")
    doc_lines.append("- Forward fill")
    doc_lines.append("")
    doc_lines.append("### Fase 4: Detección de Señales")
    doc_lines.append("- Cálculo de Global Max Bid / Min Ask")
    doc_lines.append("- Rising Edge Detection")
    doc_lines.append("")
    doc_lines.append("### Fase 5: Simulación de Latencia")
    doc_lines.append("- Time Machine")
    doc_lines.append("- Re-cálculo de profit")
    doc_lines.append("")
    doc_lines.append("### Fase 6: Análisis Final")
    doc_lines.append("- Análisis de oportunidades")
    doc_lines.append("- Estimación de ROI")
    doc_lines.append("")
    
    doc_lines.append("---")
    doc_lines.append("*Documento generado automáticamente*")
    
    # Guardar
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(doc_lines))
    
    print(f"    Documento completo guardado: {output_path}")


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


def main():
    """
    Función principal que ejecuta el pipeline completo.
    """
    
    print("=" * 80)
    print("SISTEMA DE DETECCIÓN DE ARBITRAJE HFT")
    print("Mercados Fragmentados Europeos - BME + MTFs")
    print("=" * 80)
    
    # ========================================================================
    # CONFIGURACIÓN
    # ========================================================================
    logger.info("Inicializando sistema...")
    
    # Usar DATA_SMALL por defecto (más rápido para testing)
    data_dir = config.DATA_SMALL_DIR
    
    logger.info(f"Directorio de datos: {data_dir}")
    
    # Verificar que el directorio existe
    if not data_dir.exists():
        logger.error(f"[ERROR] Directorio no encontrado: {data_dir}")
        logger.error("Por favor, asegúrate de que los datos están en la ubicación correcta")
        return
    
    # Limpiar outputs anteriores
    clean_output_directories()
    
    # Crear directorios de output (ya están limpios)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    # ========================================================================
    # FASE 1: CARGA DE DATOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("FASE 1: CARGA DE DATOS")
    print("=" * 80)
    
    loader = DataLoader(data_dir)
    
    # Descubrir ISINs disponibles
    isins = loader.discover_isins()
    
    if len(isins) == 0:
        logger.error("No se encontraron ISINs en el directorio")
        return
    
    # Seleccionar primer ISIN para análisis
    test_isin = isins[0]
    logger.info(f"Analizando ISIN: {test_isin}")
    
    # Cargar datos raw
    raw_data = loader.load_isin_data(test_isin)
    
    if len(raw_data) == 0:
        logger.error(f"No se pudieron cargar datos para {test_isin}")
        return
    
    # ========================================================================
    # FASE 2: LIMPIEZA Y VALIDACIÓN
    # ========================================================================
    print("\n" + "=" * 80)
    print("FASE 2: LIMPIEZA Y VALIDACIÓN")
    print("=" * 80)
    
    cleaner = DataCleaner()
    clean_data = cleaner.clean_all_venues(raw_data)
    
    if len(clean_data) == 0:
        logger.error("No quedan datos después de la limpieza")
        return
    
    # ========================================================================
    # FASE 3: CONSOLIDATED TAPE (OPTIMIZADO)
    # ========================================================================
    print("\n" + "=" * 80)
    print("FASE 3: CONSOLIDATED TAPE")
    print("=" * 80)
    
    # CORRECCIÓN: Usar consolidador optimizado con redondeo temporal
    tape_builder = ConsolidatedTape(time_bin_ms=100)  # 100ms bins
    
    try:
        consolidated_tape = tape_builder.create_tape(clean_data)
        
        if consolidated_tape is None or len(consolidated_tape) == 0:
            logger.error("[ERROR] Consolidated tape vacío o nulo")
            return
        
        # Validar tape
        is_valid = tape_builder.validate_tape(consolidated_tape)
        
        if not is_valid:
            logger.error("Consolidated tape falló la validación")
            return
        
        # Estadísticas del tape
        print("\n  [ESTADÍSTICAS DEL CONSOLIDATED TAPE]")
        tape_stats = tape_builder.compute_statistics(consolidated_tape)
        display_dataframe(tape_stats, "Estadísticas por Venue", max_rows=10)
        
        # Visualizar tape (se mostrará automáticamente)
        print("\n  [GENERANDO VISUALIZACIÓN DEL TAPE]")
        tape_builder.visualize_tape(consolidated_tape, test_isin)
        print("  [OK] Gráfica del tape generada y mostrada")
        
    except MemoryError as e:
        logger.error(f"[ERROR MEMORIA] No hay suficiente memoria para procesar: {e}")
        logger.error("Intenta con un dataset más pequeño o aumenta time_bin_ms")
        return
    except Exception as e:
        logger.error(f"[ERROR] Error creando consolidated tape: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return
    
    # ========================================================================
    # FASE 4: DETECCIÓN DE SEÑALES (NUEVO - PARTE 3)
    # ========================================================================
    print("\n" + "=" * 80)
    print("FASE 4: DETECCIÓN DE SEÑALES DE ARBITRAJE")
    print("=" * 80)
    
    try:
        signal_gen = SignalGenerator()
        
        # REQUISITO 1: Inicializar lista de trades ejecutados (vacía al inicio)
        executed_trades = []
        
        # Detectar oportunidades (con tracking de ejecuciones)
        print("\n  [DETECTANDO OPORTUNIDADES]")
        signals_df = signal_gen.detect_opportunities(
            consolidated_tape,
            executed_trades=executed_trades,
            isin=test_isin
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
        
        # Analizar pares de venues
        print("\n  [ANÁLISIS DE PARES DE VENUES]")
        venue_pairs = signal_gen.analyze_venue_pairs(signals_df)
        if venue_pairs is not None and len(venue_pairs) > 0:
            display_dataframe(venue_pairs, "Top Pares de Venues por Profit", max_rows=10)
        
        # Visualizar señales (se mostrará automáticamente)
        print("\n  [GENERANDO VISUALIZACIÓN DE SEÑALES]")
        signal_gen.visualize_signals(signals_df, test_isin)
        print("  [OK] Gráficas de señales generadas y mostradas")
        
        # Exportar oportunidades
        print("\n  [EXPORTANDO OPORTUNIDADES]")
        signal_gen.export_opportunities(
            signals_df,
            output_path=config.OUTPUT_DIR / f"opportunities_{test_isin}.csv"
        )
        
    except Exception as e:
        logger.error(f"[ERROR] Error en detección de señales: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        # Continuar sin señales
        signals_df = None
        venue_pairs = None
        executed_trades = []
    
    # ========================================================================
    # FASE 5: SIMULACIÓN DE LATENCIA
    # ========================================================================
    print("\n" + "=" * 80)
    print("FASE 5: SIMULACIÓN DE LATENCIA")
    print("=" * 80)
    
    exec_df = None
    try:
        if signals_df is not None and len(signals_df) > 0:
            sim = LatencySimulator(latency_us=100)
            
            # REQUISITO 1: simulate_execution ahora retorna (DataFrame, executed_trades_list)
            print("\n  [SIMULANDO EJECUCIÓN CON LATENCIA]")
            exec_df, new_executed_trades = sim.simulate_execution(
                signals_df,
                consolidated_tape,
                isin=test_isin
            )
            
            # REQUISITO 1: Actualizar lista de trades ejecutados
            executed_trades.extend(new_executed_trades)
            
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
            
            # Análisis de sensibilidad
            print("\n  [ANÁLISIS DE SENSIBILIDAD A LATENCIA]")
            sensitivity_df = sim.sensitivity_analysis(
                signals_df,
                consolidated_tape,
                latencies_us=[10, 50, 100, 200, 500, 1000]
            )
            if sensitivity_df is not None and len(sensitivity_df) > 0:
                display_dataframe(sensitivity_df, "Sensibilidad a Diferentes Latencias", max_rows=10)
            
            # Visualizar impacto de latencia (se mostrará automáticamente)
            print("\n  [GENERANDO VISUALIZACIÓN DE IMPACTO DE LATENCIA]")
            sim.visualize_latency_impact(sensitivity_df, test_isin)
            print("  [OK] Gráficas de impacto de latencia generadas y mostradas")
            
            # Exportar resultados de ejecución
            print("\n  [EXPORTANDO RESULTADOS DE EJECUCIÓN]")
            sim.export_execution_results(
                exec_df,
                output_path=config.OUTPUT_DIR / f"execution_{test_isin}.csv"
            )
        else:
            print("  [SKIP] No hay señales para simular latencia")
    except Exception as e:
        logger.error(f"[ERROR] Error en simulación de latencia: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # ========================================================================
    # FASE 6: ANÁLISIS FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("FASE 6: ANÁLISIS FINAL")
    print("=" * 80)
    
    metrics = None
    roi_metrics = None
    try:
        if signals_df is not None and exec_df is not None:
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
            
            # Generar reporte
            print("\n  [GENERANDO REPORTE FINAL]")
            report = analyzer.generate_summary_report(
                metrics,
                roi_metrics,
                output_path=config.OUTPUT_DIR / f'report_{test_isin}.txt'
            )
            print("  [OK] Reporte generado")
        else:
            print("  [SKIP] No hay datos suficientes para análisis final")
    except Exception as e:
        logger.error(f"[ERROR] Error en análisis final: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMEN DEL ANÁLISIS")
    print("=" * 80)
    
    print(f"\n  ISIN: {test_isin}")
    print(f"  - Total snapshots en tape: {len(consolidated_tape):,}")
    print(f"  - Venues incluidos: {len(clean_data)}")
    
    total_profit = 0
    if signals_df is not None and len(signals_df) > 0:
        total_opportunities = signals_df['is_rising_edge'].sum()
        total_profit = signals_df[signals_df['is_rising_edge']]['total_profit'].sum()
        
        print(f"  - Oportunidades detectadas: {total_opportunities:,}")
        print(f"  - Profit teórico total (latencia=0): €{total_profit:.2f}")
        
        if total_opportunities > 0:
            avg_profit = signals_df[signals_df['is_rising_edge']]['total_profit'].mean()
            print(f"  - Profit medio por oportunidad: €{avg_profit:.2f}")
    
    if exec_df is not None and len(exec_df) > 0:
        profitable_ops = (exec_df['profit_category'] == 'Profitable').sum()
        real_profit = exec_df['real_total_profit'].sum()
        print(f"  - Oportunidades profitable (con latencia): {profitable_ops:,}")
        print(f"  - Profit real total (con latencia): €{real_profit:.2f}")
    
    if roi_metrics:
        print(f"  - ROI estimado: {roi_metrics.get('roi_pct', 0):.4f}%")
    
    print("\n" + "=" * 80)
    print("[ÉXITO] ANÁLISIS COMPLETADO CON ÉXITO")
    print("=" * 80)
    
    # ========================================================================
    # GENERAR DOCUMENTO MARKDOWN COMPLETO
    # ========================================================================
    print("\n  [GENERANDO DOCUMENTO MARKDOWN COMPLETO]")
    generate_complete_markdown_report(
        test_isin, consolidated_tape, signals_df, exec_df, 
        venue_pairs, metrics, roi_metrics, clean_data
    )
    
    print("\n  [ARCHIVOS GENERADOS]")
    print(f"  - Log: {config.OUTPUT_DIR / 'arbitrage_system.log'}")
    print(f"  - Documento completo: {config.OUTPUT_DIR / f'complete_report_{test_isin}.md'}")
    if signals_df is not None:
        print(f"  - Oportunidades: {config.OUTPUT_DIR / f'opportunities_{test_isin}.csv'}")
    if exec_df is not None:
        print(f"  - Ejecuciones: {config.OUTPUT_DIR / f'execution_{test_isin}.csv'}")
    if metrics:
        print(f"  - Reporte: {config.OUTPUT_DIR / f'report_{test_isin}.txt'}")
    print(f"  - Figuras: {config.FIGURES_DIR}")
    
    return {
        'isin': test_isin,
        'consolidated_tape': consolidated_tape,
        'signals': signals_df,
        'executions': exec_df,
        'venue_pairs': venue_pairs,
        'metrics': metrics,
        'roi_metrics': roi_metrics,
        'total_profit': total_profit
    }


if __name__ == "__main__":
    try:
        results = main()
        if results:
            logger.info("Pipeline ejecutado exitosamente")
            print(f"\n[INFO] Resultados disponibles en variable 'results'")
            print(f"       consolidated_tape shape: {results['consolidated_tape'].shape}")
        else:
            logger.error("Pipeline falló")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}", exc_info=True)
        sys.exit(1)