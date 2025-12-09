"""
================================================================================
config_module.py - Configuración Centralizada del Sistema de Arbitraje
================================================================================

Define todos los parámetros críticos, thresholds, directorios y constantes
utilizados por los demás módulos del sistema de análisis de arbitraje.
================================================================================
"""

from pathlib import Path

class Config:
    """
    Configuración centralizada del proyecto.
    Todos los módulos importan esta clase para acceder a parámetros globales.
    """
    
    # ========================================================================
    # DIRECTORIOS
    # ========================================================================
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_SMALL_DIR = DATA_DIR / "DATA_SMALL"  # Para testing rápido
    DATA_BIG_DIR = DATA_DIR / "DATA_BIG"      # Para producción
    OUTPUT_DIR = PROJECT_ROOT / "output"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    
    # ========================================================================
    # EXCHANGES - Lista de exchanges a procesar
    # ========================================================================
    EXCHANGES = ["BME", "AQUIS", "CBOE", "TURQUOISE"]
    
    # ========================================================================
    # MAGIC NUMBERS - Vendor Data Definitions
    # ========================================================================
    # CRÍTICO: Estos valores NO son precios reales
    MAGIC_NUMBERS = [
        666666.666,  # Unquoted/Unknown
        999999.999,  # Market Order (At Best)
        999999.989,  # At Open Order
        999999.988,  # At Close Order
        999999.979,  # Pegged Order
        999999.123   # Unquoted/Unknown
    ]
    
    # ========================================================================
    # MARKET STATUS CODES - Solo Continuous Trading
    # ========================================================================
    # Estados de trading continuo válidos por exchange
    # Mapeo de exchanges a códigos de estado válidos para continuous trading
    CONTINUOUS_TRADING_STATUS = {
        'AQUIS': [5308427],
        'BME': [5832713, 5832756],
        'CBOE': [12255233],
        'TURQUOISE': [7608181]
    }
    
    # Compatibilidad con código existente (mapeo de MICs a códigos)
    VALID_STATES = {
        'XMAD': [5832713, 5832756],  # BME (Bolsas y Mercados Españoles)
        'AQXE': [5308427],            # AQUIS Exchange
        'AQEU': [5308427],            # AQUIS Exchange (variante encontrada en datos)
        'CEUX': [12255233],           # CBOE Europe
        'TRQX': [7608181],            # Turquoise
        'TQEX': [7608181]             # Turquoise (variante encontrada en datos)
    }
    
    # ========================================================================
    # LATENCY LEVELS (microsegundos)
    # ========================================================================
    # Niveles de latencia para simulación (en microsegundos)
    LATENCY_LEVELS = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 30000, 50000, 100000]
    
    # Compatibilidad con código existente
    LATENCY_BUCKETS = LATENCY_LEVELS
    
    # ========================================================================
    # THRESHOLDS DE VALIDACIÓN
    # ========================================================================
    MIN_PROFIT_PER_UNIT = 0.0001        # 0.01 céntimos
    MIN_THEORETICAL_PROFIT = 0.10     # 10 céntimos
    MAX_REASONABLE_PRICE = 10000       # EUR - Precio máximo razonable
    SUSPICIOUS_PROFIT_THRESHOLD = 1000 # EUR por ISIN
    
    # ========================================================================
    # PERFORMANCE SETTINGS
    # ========================================================================
    CHUNK_SIZE = 100000  # Filas por chunk para procesamiento
    N_JOBS = -1          # Usar todos los CPUs disponibles (-1 = auto)
    
    # ========================================================================
    # TRADING HOURS - Filtros Temporales (opcional, no usado ya que se descataloga por código de mercado)
    # ========================================================================
    MARKET_OPEN_WARMUP_SECONDS = 300    # Ignorar primeros 5 minutos
    MARKET_CLOSE_COOLDOWN_SECONDS = 300 # Ignorar últimos 5 minutos

# ============================================================================
# Instancia global para importar desde otros módulos
# ============================================================================
config = Config()
