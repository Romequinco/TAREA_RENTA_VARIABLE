"""
================================================================================
config.py - Configuración Centralizada del Sistema de Arbitraje
================================================================================
Define todos los parámetros críticos, thresholds, directorios y constantes
utilizados por los demás módulos.
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
    # MAGIC NUMBERS - Vendor Data Definitions
    # ========================================================================
    # CRÍTICO: Estos valores NO son precios reales
    # Fuente: Arbitrage study in BME.docx - Section 2 "CRITICAL"
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
    # Fuente: arbitrage_architecture.md - Sección 2.3
    # NOTA: Se incluyen variantes de códigos MIC encontradas en los datos
    VALID_STATES = {
        'XMAD': [5832713, 5832756],  # BME (Bolsas y Mercados Españoles)
        'AQXE': [5308427],            # AQUIS Exchange
        'AQEU': [5308427],            # AQUIS Exchange (variante encontrada en datos)
        'CEUX': [12255233],           # CBOE Europe
        'TRQX': [7608181],            # Turquoise
        'TQEX': [7608181]             # Turquoise (variante encontrada en datos)
    }
    
    # ========================================================================
    # LATENCY BUCKETS (microsegundos)
    # ========================================================================
    # Fuente: Arbitrage study in BME.docx - Step 4
    LATENCY_BUCKETS = [
        0,       # Teórico (instantáneo)
        100,     # 0.1ms
        500,     # 0.5ms
        1000,    # 1ms
        2000,    # 2ms
        3000,    # 3ms
        4000,    # 4ms
        5000,    # 5ms
        10000,   # 10ms
        15000,   # 15ms
        20000,   # 20ms
        30000,   # 30ms
        50000,   # 50ms
        100000   # 100ms
    ]
    
    # ========================================================================
    # TRADING HOURS - Filtros Temporales
    # ========================================================================
    # Fuente: arbitrage_architecture.md - Sección 5.3 Edge Cases
    MARKET_OPEN_WARMUP_SECONDS = 300    # Ignorar primeros 5 minutos
    MARKET_CLOSE_COOLDOWN_SECONDS = 300 # Ignorar últimos 5 minutos
    
    # ========================================================================
    # THRESHOLDS DE VALIDACIÓN
    # ========================================================================
    MIN_PROFIT_PER_UNIT = 0.0001        # 0.01 céntimos
    MIN_THEORETICAL_PROFIT = 0.10       # 10 céntimos
    MAX_REASONABLE_PRICE = 10000        # EUR
    SUSPICIOUS_PROFIT_THRESHOLD = 1000  # EUR por ISIN
    
    # ========================================================================
    # PERFORMANCE SETTINGS
    # ========================================================================
    CHUNK_SIZE = 100000  # Filas por chunk para procesamiento
    N_JOBS = -1          # Usar todos los CPUs disponibles (-1 = auto)
    
    # ========================================================================
    # BOOK IDENTITY KEY
    # ========================================================================
    # Fuente: arbitrage_architecture.md - Sección 2.1
    # book_key = (session, isin, mic, ticker)
    # Esto identifica unívocamente cada order book para joins QTE-STS

# ============================================================================
# Instancia global para importar desde otros módulos
# ============================================================================
config = Config()
