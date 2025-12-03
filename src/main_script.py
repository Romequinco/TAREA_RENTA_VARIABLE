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
from pathlib import Path

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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_DIR / 'arbitrage_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


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
    
    # Crear directorios de output
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
        tape_stats = tape_builder.compute_statistics(consolidated_tape)
        
        # Visualizar tape
        tape_builder.visualize_tape(consolidated_tape, test_isin)
        
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
        
        # Detectar oportunidades
        signals_df = signal_gen.detect_opportunities(consolidated_tape)
        
        # Analizar pares de venues
        venue_pairs = signal_gen.analyze_venue_pairs(signals_df)
        
        # Visualizar señales
        signal_gen.visualize_signals(signals_df, test_isin)
        
        # Exportar oportunidades
        signal_gen.export_opportunities(signals_df)
        
    except Exception as e:
        logger.error(f"[ERROR] Error en detección de señales: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        # Continuar sin señales
        signals_df = None
        venue_pairs = None
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMEN DEL ANÁLISIS")
    print("=" * 80)
    
    print(f"\nISIN: {test_isin}")
    print(f"  - Total snapshots en tape: {len(consolidated_tape):,}")
    print(f"  - Venues incluidos: {len(clean_data)}")
    
    if signals_df is not None and len(signals_df) > 0:
        total_opportunities = signals_df['is_rising_edge'].sum()
        total_profit = signals_df[signals_df['is_rising_edge']]['total_profit'].sum()
        
        print(f"  - Oportunidades detectadas: {total_opportunities:,}")
        print(f"  - Profit teórico total (latencia=0): €{total_profit:.2f}")
        
        if total_opportunities > 0:
            avg_profit = signals_df[signals_df['is_rising_edge']]['total_profit'].mean()
            print(f"  - Profit medio por oportunidad: €{avg_profit:.2f}")
    else:
        print(f"  - [INFO] No se ejecutó detección de señales")
    
    print("\n" + "=" * 80)
    print("[EXITO] ANÁLISIS COMPLETADO CON ÉXITO")
    print("=" * 80)
    
    print("\nArchivos generados:")
    print(f"  - Log: {config.OUTPUT_DIR / 'arbitrage_system.log'}")
    if signals_df is not None:
        print(f"  - Oportunidades: {config.OUTPUT_DIR / 'opportunities.csv'}")
    
    print("\n[INFO] Próximos pasos:")
    print("  - PARTE 4: Simulación de Latencia (Time Machine)")
    print("  - PARTE 5: Análisis Final + Reporte Completo")
    
    return {
        'isin': test_isin,
        'consolidated_tape': consolidated_tape,
        'signals': signals_df,
        'venue_pairs': venue_pairs,
        'total_profit': total_profit if signals_df is not None else 0
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