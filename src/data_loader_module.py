"""
================================================================================
data_loader.py - Módulo de Carga de Archivos QTE y STS (OPTIMIZADO)
================================================================================

ESPECIFICACIONES TÉCNICAS:

**Naming Convention de Archivos:**
<type>_<session>_<isin>_<ticker>_<mic>_<part>.csv.gz

Donde:
- type: QTE (quotes) o STS (status)
- session: YYYY-MM-DD
- isin: Código internacional del instrumento
- mic: Código de venue (XMAD, AQXE, CEUX, TRQX)
- ticker: Símbolo específico del venue
- part: Siempre 1

**Book Identity Key:** (session, isin, mic, ticker)

**REQUISITOS DEL CÓDIGO:**
1. Función principal: load_data_for_isin(session, isin)
2. Columnas requeridas en QTE: epoch (int64), px_bid_0, px_ask_0, qty_bid_0, qty_ask_0
3. Columnas requeridas en STS: epoch (int64), market_trading_status (int64)
4. Forzar dtype={'epoch': 'int64'} para evitar pérdida de precisión
5. Manejo robusto de archivos faltantes y corruptos
6. Validación post-carga completa
7. Soporte para paralelización opcional

EDGE CASES MANEJADOS:
- Archivo corrupto → Skip y log warning
- Venue sin datos para ese ISIN → Retornar None para ese MIC
- Archivo vacío → Skip y log warning
- Session/ISIN mismatch → Skip y log warning

OPTIMIZACIONES APLICADAS:
- Consolidación de código duplicado entre load_qte_file y load_sts_file
- Métodos compartidos: _read_csv_with_encoding, _convert_epoch_to_int64, _validate_gzip_file
- Validaciones optimizadas con early exit y operaciones vectorizadas
- Parsing de filenames optimizado con método estático reutilizable
- Conversión de tipos vectorizada para precios/cantidades
- Filtros combinados en una sola pasada para mejor rendimiento
- Paralelización mejorada con chunks y imap para mejor balanceo de carga
- Reducción de copias innecesarias de DataFrames
- Uso de sets para validación de columnas (O(1) vs O(n))

COMPATIBILIDAD:
- Mantiene compatibilidad con load_isin_data() usado en main_script.py y main_big_data.py
- Todas las funciones existentes siguen funcionando igual

================================================================================
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import sys
import gzip
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

# Constantes para optimización
ENCODINGS = ['utf-8', 'latin-1', 'cp1252']
CSV_READ_KWARGS = {
    'compression': 'gzip',
    'sep': ';',
    'decimal': '.',
    'low_memory': False
}


# ============================================================================
# FUNCIÓN PRINCIPAL (Wrapper estático para compatibilidad)
# ============================================================================

def load_data_for_isin(data_dir: Path, session: str, isin: str) -> Dict[str, Dict]:
    """
    Función principal para cargar datos de un ISIN específico en una sesión específica.
    
    Esta es la función wrapper estática que cumple con las especificaciones.
    Internamente usa la clase DataLoader.
    
    Args:
        data_dir: Path al directorio conteniendo archivos .csv.gz
        session: Fecha de sesión en formato YYYY-MM-DD
        isin: International Securities Identification Number
        
    Returns:
        Diccionario con estructura:
        {
            'XMAD': {'qte': DataFrame, 'sts': DataFrame},
            'AQXE': {'qte': DataFrame, 'sts': DataFrame},
            ...
        }
        Si un venue no está disponible, no aparecerá en el diccionario.
        
    Ejemplo:
        >>> from pathlib import Path
        >>> data = load_data_for_isin(Path('data'), '2024-01-15', 'ES0113900J37')
        >>> print(data['XMAD']['qte'].head())
    """
    loader = DataLoader(data_dir)
    return loader.load_data_for_isin(session, isin)


class DataLoader:
    """
    Clase responsable de la ingesta de datos desde archivos comprimidos.
    
    IMPORTANTE: Los archivos CSV tienen:
    - Separador: punto y coma (;)
    - Decimal: punto (.) - formato anglosajón estándar
    - Encoding: UTF-8 o Latin-1
    """
    
    # Columnas requeridas (nombres exactos en los archivos)
    QTE_REQUIRED = ['epoch', 'px_bid_0', 'px_ask_0', 'qty_bid_0', 'qty_ask_0']
    STS_REQUIRED = ['epoch', 'market_trading_status']
    
    # Columnas opcionales (niveles 1-9 si existen)
    QTE_OPTIONAL_LEVELS = []
    for level in range(1, 10):
        QTE_OPTIONAL_LEVELS.extend([
            f'px_bid_{level}', f'px_ask_{level}',
            f'qty_bid_{level}', f'qty_ask_{level}'
        ])
    
    def __init__(self, data_dir: Path):
        """
        Inicializa el loader con el directorio de datos.
        
        Args:
            data_dir: Path al directorio conteniendo subcarpetas con archivos .csv.gz
        """
        self.data_dir = data_dir
        
        # Configurar encoding para Windows
        if sys.platform == 'win32':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except:
                pass
        
        logger.info(f"DataLoader initialized: {data_dir}")
    
    @staticmethod
    def _extract_isin_from_path(filepath: str) -> Optional[str]:
        """Extrae ISIN del path del archivo. Optimizado para reutilización."""
        parts = Path(filepath).stem.split('_')
        return parts[2] if len(parts) >= 5 else None
    
    def discover_isins(self) -> List[str]:
        """
        Descubre todos los ISINs únicos disponibles en el directorio.
        Optimizado con set comprehension.
        """
        print("\n" + "=" * 80)
        print("DESCUBRIENDO ISINs DISPONIBLES")
        print("=" * 80)
        
        # Búsqueda recursiva optimizada
        qte_files = glob.glob(str(self.data_dir / "**" / "QTE_*.csv.gz"), recursive=True)
        
        # Extraer ISINs usando set comprehension (más eficiente)
        isins = {self._extract_isin_from_path(f) for f in qte_files}
        isins.discard(None)  # Eliminar None si hay archivos con formato inválido
        
        isins_list = sorted(isins)
        
        print(f" Encontrados {len(isins_list)} ISINs unicos")
        if len(isins_list) > 0:
            print(f"  Primeros 5: {isins_list[:5]}")
        
        return isins_list
    
    def _validate_gzip_file(self, filepath: Path) -> bool:
        """Valida que un archivo gzip no está corrupto."""
        try:
            with gzip.open(filepath, 'rb') as f:
                f.read(1)
            return True
        except (gzip.BadGzipFile, OSError) as e:
            logger.error(f"  [ERROR] Archivo corrupto (gzip): {filepath.name} - {e}")
            return False
    
    def _read_csv_with_encoding(self, filepath: Path, required_cols: List[str], 
                                force_epoch_int64: bool = True) -> Optional[pd.DataFrame]:
        """
        Lee un CSV con múltiples encodings y valida columnas requeridas.
        Optimizado para reducir duplicación de código.
        """
        if not filepath.exists():
            logger.error(f"  [ERROR] Archivo no existe: {filepath.name}")
            return None
        
        if not self._validate_gzip_file(filepath):
            return None
        
        # Intentar leer con diferentes encodings
        read_kwargs = CSV_READ_KWARGS.copy()
        if force_epoch_int64:
            read_kwargs['dtype'] = {'epoch': 'int64'}
        
        for encoding in ENCODINGS:
            try:
                read_kwargs['encoding'] = encoding
                df = pd.read_csv(filepath, **read_kwargs)
                
                # Limpiar nombres de columnas
                df.columns = df.columns.str.strip()
                
                # Validar columnas requeridas
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"  [ERROR] Faltan columnas en {filepath.name}: {missing_cols}")
                    return None
                
                # Validar archivo vacío
                if len(df) == 0:
                    logger.warning(f"  [ADVERTENCIA] Archivo vacío: {filepath.name}")
                    return None
                
                logger.debug(f"  Archivo leido correctamente con encoding: {encoding}")
                return df
                
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
            except ValueError as e:
                # Si epoch no es convertible directamente, intentar sin dtype
                if force_epoch_int64 and 'epoch' in str(e).lower():
                    try:
                        read_kwargs_no_dtype = CSV_READ_KWARGS.copy()
                        read_kwargs_no_dtype['encoding'] = encoding
                        df = pd.read_csv(filepath, **read_kwargs_no_dtype)
                        df.columns = df.columns.str.strip()
                        if all(col in df.columns for col in required_cols) and len(df) > 0:
                            return df
                    except:
                        continue
                continue
            except Exception as e:
                logger.debug(f"  Error con encoding {encoding}: {e}")
                continue
        
        logger.error(f"  [ERROR] No se pudo leer {filepath.name} con ningun encoding")
        return None
    
    def _convert_epoch_to_int64(self, df: pd.DataFrame, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Convierte epoch a int64 con validación.
        
        CRÍTICO: epoch debe ser int64 (nanosegundos UTC) para evitar pérdida de precisión
        y errores en operaciones temporales (merge_asof, búsquedas binarias, etc.).
        
        Args:
            df: DataFrame con columna 'epoch'
            filepath: Path del archivo (para logging)
            
        Returns:
            DataFrame con epoch convertido a int64, o None si falla la conversión
        """
        # Early exit: si ya es int64, no hacer nada
        if df['epoch'].dtype == 'int64':
            return df
        
        # Eliminar NaNs primero (no se pueden convertir)
        rows_before = len(df)
        df = df.dropna(subset=['epoch']).copy()  # Copiar solo si necesitamos modificar
        removed_nans = rows_before - len(df)
        if removed_nans > 0:
            logger.warning(f"  Eliminadas {removed_nans} filas con epoch NaN")
        
        if len(df) == 0:
            logger.error(f"  [ERROR] No quedan filas después de eliminar NaNs en {filepath.name}")
            return None
        
        try:
            # Convertir a numérico primero (maneja strings, floats, etc.)
            df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
            
            # Eliminar valores que no se pudieron convertir (NaNs)
            df = df.dropna(subset=['epoch'])
            
            if len(df) == 0:
                logger.error(f"  [ERROR] No quedan filas válidas después de conversión numérica en {filepath.name}")
                return None
            
            # Convertir a int64 (puede fallar si hay valores fuera de rango)
            df['epoch'] = df['epoch'].astype('int64')
            
            # Verificación final (no debería haber NaNs después de astype('int64'))
            if df['epoch'].isna().any():
                logger.error(f"  [ERROR] Epoch contiene NaNs después de conversión en {filepath.name}")
                return None
            
            return df
            
        except (ValueError, OverflowError) as e:
            logger.error(f"  [ERROR] No se pudo convertir epoch a int64 en {filepath.name}: {e}")
            return None
    
    def load_qte_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Carga un archivo QTE individual (order book snapshots).
        Optimizado para mejor rendimiento.
        """
        try:
            # Leer CSV con validación
            df = self._read_csv_with_encoding(filepath, self.QTE_REQUIRED, force_epoch_int64=True)
            if df is None:
                return None
            
            # Convertir epoch a int64 si es necesario
            df = self._convert_epoch_to_int64(df, filepath)
            if df is None:
                return None
            
            # Convertir precios y cantidades a float64 (optimizado: vectorizado)
            # Generar lista de columnas de precios/cantidades (niveles 0-9)
            price_qty_cols = [
                f'{prefix}{level}' 
                for level in range(10) 
                for prefix in ['px_bid_', 'px_ask_', 'qty_bid_', 'qty_ask_']
                if f'{prefix}{level}' in df.columns
            ]
            
            # Convertir todas las columnas de una vez (más eficiente que columna por columna)
            if price_qty_cols:
                df[price_qty_cols] = df[price_qty_cols].apply(
                    pd.to_numeric, errors='coerce', downcast='float'
                ).astype('float64')
            
            # Validaciones y filtros (optimizado: una sola pasada)
            initial_len = len(df)
            
            # Filtro combinado: precios > 0 y spread válido
            mask = (
                (df['px_bid_0'] > 0) & 
                (df['px_ask_0'] > 0) &
                (df['qty_bid_0'] > 0) &
                (df['qty_ask_0'] > 0) &
                (df['px_ask_0'] >= df['px_bid_0'])
            )
            
            invalid_count = (~mask).sum()
            if invalid_count > 0:
                if (df['px_ask_0'] < df['px_bid_0']).any():
                    logger.warning(f"  [ADVERTENCIA] {invalid_count} filas con datos inválidos")
                df = df[mask]
            
            logger.info(f"  [OK] {filepath.name}: {len(df):,} filas válidas (de {initial_len:,})")
            return df
        
        except Exception as e:
            logger.error(f"  [ERROR] Error inesperado al cargar {filepath.name}: {str(e)}")
            logger.debug(f"    Tipo: {type(e).__name__}", exc_info=True)
            return None
    
    def load_sts_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Carga un archivo STS individual (trading status changes).
        Optimizado usando métodos compartidos.
        """
        try:
            # Leer CSV con validación
            df = self._read_csv_with_encoding(filepath, self.STS_REQUIRED, force_epoch_int64=True)
            if df is None:
                return None
            
            # Convertir epoch a int64 si es necesario
            df = self._convert_epoch_to_int64(df, filepath)
            if df is None:
                return None
            
            # Convertir market_trading_status
            df['market_trading_status'] = pd.to_numeric(
                df['market_trading_status'], errors='coerce'
            ).astype('Int64')
            
            logger.info(f"  [OK] {filepath.name}: {len(df):,} filas")
            return df
        
        except Exception as e:
            logger.error(f"  [ERROR] Error al cargar {filepath.name}: {str(e)}")
            logger.debug(f"    Tipo: {type(e).__name__}", exc_info=True)
            return None
    
    def inspect_file_structure(self, filepath: Path):
        """
        Inspecciona la estructura de un archivo para debugging.
        
        Args:
            filepath: Path al archivo a inspeccionar
        """
        print(f"\n[INSPECCION] {filepath.name}")
        print("=" * 80)
        
        try:
            # Intentar leer primeras filas
            df = None
            encoding_used = None
            
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(
                        filepath, 
                        compression='gzip', 
                        nrows=5,
                        sep=';',  # Separador: punto y coma
                        decimal='.',  # Decimal anglosajón (punto)
                        # NO especificar 'thousands' - los datos usan formato estándar
                        encoding=encoding,
                        low_memory=False
                    )
                    encoding_used = encoding
                    break
                except:
                    continue
            
            if df is None:
                print("[ERROR] No se pudo leer el archivo")
                return
            
            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip()
            
            print(f"Encoding usado: {encoding_used}")
            print(f"Total columnas: {len(df.columns)}")
            print(f"Total filas (muestra): {len(df)}")
            
            print(f"\nColumnas requeridas:")
            for col in self.QTE_REQUIRED:
                status = "[OK]" if col in df.columns else "[FALTA]"
                print(f"  {status} {col}")
            
            print(f"\nPrimeras 5 columnas:")
            print(list(df.columns[:5]))
            
            print(f"\nPrimeras 3 filas (columnas relevantes):")
            available_cols = [col for col in self.QTE_REQUIRED if col in df.columns]
            if available_cols:
                print(df[available_cols].head(3).to_string())
            
            print(f"\nTipos de datos:")
            if available_cols:
                print(df[available_cols].dtypes)
            
        except Exception as e:
            print(f"[ERROR] al inspeccionar: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def load_isin_data(self, isin: str) -> Dict[str, Dict]:
        """
        Carga TODOS los datos (QTE + STS) para un ISIN específico.
        MANTIENE COMPATIBILIDAD con código existente.
        Optimizado para mejor rendimiento.
        
        Args:
            isin: International Securities Identification Number
            
        Returns:
            Diccionario con estructura:
            {
                'XMAD': {'qte': DataFrame, 'sts': DataFrame},
                'AQXE': {'qte': DataFrame, 'sts': DataFrame},
                ...
            }
        """
        print("\n" + "=" * 80)
        print(f"CARGANDO DATOS PARA ISIN: {isin}")
        print("=" * 80)
        
        # Buscar archivos optimizado (solo QTE, STS se busca después)
        qte_pattern = f"QTE_*_{isin}_*.csv.gz"
        qte_files = glob.glob(str(self.data_dir / "**" / qte_pattern), recursive=True)
        
        print(f"  Archivos QTE encontrados: {len(qte_files)}")
        
        if not qte_files:
            logger.error(f"No se encontraron archivos para ISIN={isin}")
            return {}
        
        # Inspección opcional del primer archivo (solo si hay archivos)
        if logger.isEnabledFor(logging.DEBUG) and qte_files:
            self.inspect_file_structure(Path(qte_files[0]))
        
        venue_data = {}
        
        # Procesar archivos optimizado
        for qte_file in qte_files:
            qte_path = Path(qte_file)
            
            # Parsear filename de forma eficiente
            parsed = self._parse_filename(qte_path)
            if not parsed:
                logger.warning(f"  [SKIP] Formato inválido: {qte_path.name}")
                continue
            
            session, file_isin, ticker, mic, part = parsed
            
            # Validar ISIN
            if file_isin != isin:
                continue
            
            print(f"\n  [PROCESANDO] Venue: {mic}")
            
            # Cargar QTE
            qte_df = self.load_qte_file(qte_path)
            if qte_df is None or len(qte_df) == 0:
                logger.warning(f"  [SKIP] {mic} - QTE no disponible")
                continue
            
            # Buscar y cargar STS
            sts_file = qte_path.parent / qte_path.name.replace('QTE_', 'STS_')
            sts_df = None
            
            if sts_file.exists():
                sts_df = self.load_sts_file(sts_file)
            
            # Añadir Book Identity Key columns (optimizado)
            qte_df = self._add_identity_columns(qte_df, session, isin, ticker, mic)
            if sts_df is not None:
                sts_df = self._add_identity_columns(sts_df, session, isin, ticker, mic)
            
            # Validación post-carga
            if not self._validate_post_load(qte_df, sts_df, mic):
                continue
            
            venue_data[mic] = {'qte': qte_df, 'sts': sts_df}
            print(f"    [OK] {mic}: {len(qte_df):,} snapshots")
        
        print(f"\n[EXITO] Venues cargados: {list(venue_data.keys())}")
        return venue_data
    
    def get_book_identity(self, filepath: Path) -> tuple:
        """
        Extrae el Book Identity Key del filename.
        
        Args:
            filepath: Path al archivo
            
        Returns:
            Tuple (session, isin, mic, ticker) o (None, None, None, None)
        """
        parts = Path(filepath).stem.split('_')
        
        if len(parts) >= 5:
            session = parts[1]  # Trading date
            isin = parts[2]     # ISIN
            ticker = parts[3]   # Venue-specific ticker
            mic = parts[4]      # Market Identifier Code
            return (session, isin, mic, ticker)
        
        return (None, None, None, None)
    
    @staticmethod
    def _parse_filename(filepath: Path) -> Optional[Tuple[str, str, str, str, str]]:
        """Parsea filename y retorna (session, isin, ticker, mic, part) o None."""
        parts = filepath.stem.split('_')
        if len(parts) >= 6:
            return tuple(parts[1:6])  # session, isin, ticker, mic, part
        return None
    
    def _add_identity_columns(self, df: pd.DataFrame, session: str, isin: str, 
                             ticker: str, mic: str) -> pd.DataFrame:
        """Añade columnas de Book Identity Key de forma eficiente."""
        if 'session' not in df.columns:
            df = df.copy()  # Solo copiar si es necesario
            df['session'] = session
            df['isin'] = isin
            df['ticker'] = ticker
            df['mic'] = mic
        return df
    
    def load_data_for_isin(self, session: str, isin: str) -> Dict[str, Dict]:
        """
        Función principal para cargar datos de un ISIN específico en una sesión específica.
        Optimizado para mejor rendimiento y menor uso de memoria.
        """
        logger.info(f"Cargando datos para ISIN={isin}, Session={session}")
        print("\n" + "=" * 80)
        print(f"CARGANDO DATOS PARA ISIN: {isin} | SESSION: {session}")
        print("=" * 80)
        
        venue_data = {}
        qte_count = 0
        sts_count = 0
        
        # Buscar archivos QTE optimizado
        qte_pattern = f"QTE_{session}_{isin}_*_*.csv.gz"
        qte_files = glob.glob(str(self.data_dir / "**" / qte_pattern), recursive=True)
        
        logger.info(f"Archivos QTE encontrados: {len(qte_files)}")
        print(f"  Archivos QTE encontrados: {len(qte_files)}")
        
        if not qte_files:
            logger.warning(f"No se encontraron archivos QTE para ISIN={isin}, Session={session}")
            print(f"  [ADVERTENCIA] No se encontraron archivos QTE")
            return venue_data
        
        # Procesar cada archivo QTE (optimizado)
        for qte_file in qte_files:
            qte_path = Path(qte_file)
            
            # Parsear filename de forma eficiente
            parsed = self._parse_filename(qte_path)
            if not parsed:
                logger.warning(f"  [SKIP] Formato inválido: {qte_path.name}")
                continue
            
            file_session, file_isin, ticker, mic, part = parsed
            
            # Validar Book Identity Key (early exit)
            if file_session != session or file_isin != isin:
                logger.warning(f"  [SKIP] Book Identity mismatch: session={file_session}, isin={file_isin}")
                continue
            
            print(f"\n  [PROCESANDO] Venue: {mic}")
            
            # Cargar QTE
            try:
                qte_df = self.load_qte_file(qte_path)
                if qte_df is None or len(qte_df) == 0:
                    continue
                
                qte_count += 1
                qte_df = self._add_identity_columns(qte_df, session, isin, ticker, mic)
                
            except Exception as e:
                logger.error(f"  [ERROR] Error cargando QTE para {mic}: {e}", exc_info=True)
                continue
            
            # Buscar y cargar STS
            sts_file = qte_path.parent / qte_path.name.replace('QTE_', 'STS_')
            sts_df = None
            
            if sts_file.exists():
                try:
                    sts_df = self.load_sts_file(sts_file)
                    if sts_df is not None and len(sts_df) > 0:
                        sts_count += 1
                        sts_df = self._add_identity_columns(sts_df, session, isin, ticker, mic)
                except Exception as e:
                    logger.error(f"  [ERROR] Error cargando STS para {mic}: {e}", exc_info=True)
            
            # Validación post-carga
            if not self._validate_post_load(qte_df, sts_df, mic):
                continue
            
            # Almacenar datos
            venue_data[mic] = {'qte': qte_df, 'sts': sts_df}
            print(f"    [OK] {mic}: {len(qte_df):,} QTE, "
                  f"{len(sts_df):,} STS" if sts_df is not None else "STS no disponible")
        
        # Resumen
        logger.info(f"Carga completada: {qte_count} QTE, {sts_count} STS, {len(venue_data)} venues")
        print(f"\n[EXITO] QTE: {qte_count} | STS: {sts_count} | Venues: {list(venue_data.keys())}")
        
        return venue_data
    
    def _validate_post_load(self, qte_df: pd.DataFrame, sts_df: Optional[pd.DataFrame], mic: str) -> bool:
        """
        Validación post-carga optimizada.
        Early exit para mejor rendimiento.
        """
        # Validar QTE (early exit)
        if qte_df is None or len(qte_df) == 0:
            logger.error(f"  [VALIDACION] {mic}: QTE vacío")
            return False
        
        # Verificar columnas requeridas (optimizado: usar set)
        if not set(self.QTE_REQUIRED).issubset(qte_df.columns):
            missing = set(self.QTE_REQUIRED) - set(qte_df.columns)
            logger.error(f"  [VALIDACION] {mic}: Faltan columnas QTE: {missing}")
            return False
        
        # Verificar epoch (optimizado: una sola verificación)
        if qte_df['epoch'].dtype != 'int64' or qte_df['epoch'].isna().any():
            logger.error(f"  [VALIDACION] {mic}: Epoch inválido (tipo: {qte_df['epoch'].dtype})")
            return False
        
        # Validar STS si existe (solo warnings, no bloquea)
        if sts_df is not None and len(sts_df) > 0:
            if not set(self.STS_REQUIRED).issubset(sts_df.columns):
                missing = set(self.STS_REQUIRED) - set(sts_df.columns)
                logger.warning(f"  [VALIDACION] {mic}: Faltan columnas STS: {missing}")
            
            if sts_df['epoch'].dtype != 'int64' or sts_df['epoch'].isna().any():
                logger.warning(f"  [VALIDACION] {mic}: Epoch STS inválido")
        
        return True
    
    @staticmethod
    def load_multiple_isins_parallel(data_dir: Path, sessions: List[str], isins: List[str], 
                                     n_jobs: Optional[int] = None) -> Dict[Tuple[str, str], Dict[str, Dict]]:
        """
        Carga múltiples ISINs en paralelo usando multiprocessing.
        Optimizado para mejor rendimiento con chunks.
        
        Args:
            data_dir: Directorio de datos
            sessions: Lista de sesiones (YYYY-MM-DD)
            isins: Lista de ISINs
            n_jobs: Número de procesos paralelos (None = usar todos los CPUs)
            
        Returns:
            Dict[(session, isin)] -> Dict[mic] -> {'qte': DataFrame, 'sts': DataFrame}
        """
        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)  # Dejar 1 CPU libre
        
        total_tasks = len(sessions) * len(isins)
        logger.info(f"Cargando {total_tasks} tareas usando {n_jobs} procesos")
        
        # Crear lista de tareas
        tasks = [(session, isin) for session in sessions for isin in isins]
        
        # Función helper optimizada para multiprocessing
        def _load_single_task(args):
            session, isin = args
            try:
                loader = DataLoader(data_dir)
                return (session, isin), loader.load_data_for_isin(session, isin)
            except Exception as e:
                logger.error(f"Error cargando {isin} en {session}: {e}", exc_info=True)
                return (session, isin), {}
        
        # Ejecutar en paralelo con chunks para mejor balanceo de carga
        results = {}
        chunk_size = max(1, total_tasks // (n_jobs * 4))  # 4 chunks por worker
        
        with Pool(processes=n_jobs) as pool:
            # Usar imap para mejor control de memoria
            for (session, isin), data in pool.imap(_load_single_task, tasks, chunksize=chunk_size):
                results[(session, isin)] = data
        
        logger.info(f"Carga paralela completada: {len(results)}/{total_tasks} exitosas")
        return results