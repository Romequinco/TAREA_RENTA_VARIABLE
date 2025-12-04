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
from pathlib import Path
import pandas as pd
from datetime import datetime

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
        
        # Estadísticas y visualización
        tape_stats = tape_builder.compute_statistics(consolidated_tape)
        tape_builder.visualize_tape(consolidated_tape, isin)
        
        # ====================================================================
        # FASE 4: DETECCIÓN DE SEÑALES
        # ====================================================================
        print(f"\n[FASE 4] Detectando señales para {isin}...")
        signal_gen = SignalGenerator()
        
        signals_df = signal_gen.detect_opportunities(consolidated_tape)
        venue_pairs = signal_gen.analyze_venue_pairs(signals_df)
        signal_gen.visualize_signals(signals_df, isin)
        signal_gen.export_opportunities(
            signals_df, 
            output_path=config.OUTPUT_DIR / f"opportunities_{isin}.csv"
        )
        
        # ====================================================================
        # FASE 5: SIMULACIÓN DE LATENCIA
        # ====================================================================
        print(f"\n[FASE 5] Simulando latencia para {isin}...")
        sim = LatencySimulator(latency_us=100)
        exec_df = sim.simulate_execution(signals_df, consolidated_tape)
        
        sensitivity_df = sim.sensitivity_analysis(
            signals_df, 
            consolidated_tape,
            latencies_us=[10, 50, 100, 200, 500, 1000]
        )
        
        sim.visualize_latency_impact(sensitivity_df, isin)
        sim.export_execution_results(
            exec_df,
            output_path=config.OUTPUT_DIR / f"execution_{isin}.csv"
        )
        
        # ====================================================================
        # FASE 6: ANÁLISIS FINAL
        # ====================================================================
        print(f"\n[FASE 6] Análisis final para {isin}...")
        analyzer = ArbitrageAnalyzer()
        
        metrics = analyzer.analyze_opportunities(signals_df, exec_df)
        roi_metrics = analyzer.estimate_roi(metrics, trading_costs_bps=0.5, capital_eur=100000)
        
        report = analyzer.generate_summary_report(
            metrics,
            roi_metrics,
            output_path=config.OUTPUT_DIR / f'report_{isin}.txt'
        )
        
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
        
        print(f"\n[ÉXITO] {isin} procesado correctamente")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"[ERROR] Error procesando {isin}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return results


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
    
    # Crear directorios de output
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
    # REPORTE CONSOLIDADO
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
        total_profit = sum(r['metrics']['real_profit'] for r in successful)
        total_opportunities = sum(r['metrics']['opportunities'] for r in successful)
        
        print(f"\n  Total oportunidades detectadas: {total_opportunities:,}")
        print(f"  Profit real total: €{total_profit:.2f}")
    
    print(f"\nArchivos generados:")
    print(f"  - Log: {config.OUTPUT_DIR / 'arbitrage_system_BIG.log'}")
    print(f"  - Reporte consolidado: {consolidated_report_path}")
    print(f"  - Reportes individuales: {config.OUTPUT_DIR / 'report_*.txt'}")
    print(f"  - Figuras: {config.FIGURES_DIR}")
    
    # Crear resumen CSV
    summary_data = []
    for r in successful:
        summary_data.append({
            'ISIN': r['isin'],
            'Snapshots': r['metrics']['total_snapshots'],
            'Venues': r['metrics']['venues'],
            'Opportunities': r['metrics']['opportunities'],
            'Profitable_Ops': r['metrics']['profitable_ops'],
            'Theoretical_Profit_EUR': r['metrics']['theoretical_profit'],
            'Real_Profit_EUR': r['metrics']['real_profit'],
            'ROI_PCT': r['metrics']['roi_pct']
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
