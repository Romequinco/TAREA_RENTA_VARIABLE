"""
================================================================================
data_loader.py - Módulo de Carga de Archivos QTE y STS 
================================================================================

Correcciones principales:
1. Separador CSV correcto: punto y coma (;)
2. Decimal: punto (.) - formato anglosajón estándar 
3. Eliminación de emojis Unicode para compatibilidad Windows
4. Mejor manejo de errores y encoding
5. Validación robusta de datos

================================================================================
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List, Optional
import logging
import sys

logger = logging.getLogger(__name__)


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
    
    def discover_isins(self) -> List[str]:
        """
        Descubre todos los ISINs únicos disponibles en el directorio.
        
        Returns:
            Lista ordenada de ISINs (strings)
        """
        print("\n" + "=" * 80)
        print("DESCUBRIENDO ISINs DISPONIBLES")
        print("=" * 80)
        
        # Búsqueda recursiva de archivos QTE
        qte_files = glob.glob(
            str(self.data_dir / "**" / "QTE_*.csv.gz"), 
            recursive=True
        )
        
        # Extraer ISINs del filename
        isins = set()
        for file in qte_files:
            # Formato: QTE_session_isin_ticker_mic_part.csv.gz
            parts = Path(file).stem.split('_')
            if len(parts) >= 5:
                isin = parts[2]  # Posición del ISIN
                isins.add(isin)
        
        isins_list = sorted(list(isins))
        
        print(f" Encontrados {len(isins_list)} ISINs unicos")
        if len(isins_list) > 0:
            print(f"  Primeros 5: {isins_list[:5]}")
        
        return isins_list
    
    def load_qte_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Carga un archivo QTE individual (order book snapshots).
        
        CRÍTICO: 
        - Separador: punto y coma (;)
        - Decimal: punto (.) - formato estándar anglosajón
        
        Args:
            filepath: Path completo al archivo .csv.gz
            
        Returns:
            DataFrame con los snapshots, o None si hay error
        """
        try:
            # PASO 1: Intentar leer con diferentes encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(
                        filepath, 
                        compression='gzip',
                        sep=';',  # Separador: punto y coma
                        decimal='.',  # CORRECCIÓN: Decimal anglosajón (punto)
                        # NO especificar 'thousands' - los datos usan formato estándar
                        encoding=encoding,
                        low_memory=False
                    )
                    logger.info(f"  Archivo leido correctamente con encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error(f"  [ERROR] No se pudo leer {filepath.name} con ningun encoding")
                return None
            
            # PASO 2: Limpiar nombres de columnas (quitar espacios)
            df.columns = df.columns.str.strip()
            
            # DEBUG: Mostrar columnas disponibles
            logger.info(f"  Columnas detectadas: {len(df.columns)}")
            logger.debug(f"  Primeras 10 columnas: {list(df.columns[:10])}")
            
            # PASO 3: Validar columnas requeridas
            missing_cols = [col for col in self.QTE_REQUIRED if col not in df.columns]
            
            if missing_cols:
                logger.error(f"  [ERROR] Faltan columnas en {filepath.name}: {missing_cols}")
                logger.error(f"    Columnas disponibles: {list(df.columns[:20])}")
                return None
            
            # PASO 4: Convertir tipos de datos
            # Epoch a int64 (puede venir en notación científica)
            df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
            df['epoch'] = (df['epoch'] * 1e9).astype('int64')  # Convertir a nanosegundos si es necesario
            
            # Precios y cantidades a float
            for col in ['px_bid_0', 'px_ask_0', 'qty_bid_0', 'qty_ask_0']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # DEBUG: Verificar rango de precios
            logger.info(f"  Rango de precios BID: {df['px_bid_0'].min():.4f} - {df['px_bid_0'].max():.4f}")
            logger.info(f"  Rango de precios ASK: {df['px_ask_0'].min():.4f} - {df['px_ask_0'].max():.4f}")
            
            # PASO 5: Limpiar datos
            # Eliminar filas con NaN en columnas críticas
            rows_before = len(df)
            df = df.dropna(subset=self.QTE_REQUIRED)
            rows_after = len(df)
            
            if rows_before > rows_after:
                logger.warning(f"  Eliminadas {rows_before - rows_after} filas con NaN")
            
            # Eliminar precios negativos o cero
            df = df[
                (df['px_bid_0'] > 0) & 
                (df['px_ask_0'] > 0) &
                (df['qty_bid_0'] > 0) &
                (df['qty_ask_0'] > 0)
            ]
            
            # Validar spread (ask >= bid)
            invalid_spreads = df[df['px_ask_0'] < df['px_bid_0']]
            if len(invalid_spreads) > 0:
                logger.warning(f"  [ADVERTENCIA] {len(invalid_spreads)} filas con spread invalido (ask < bid)")
                df = df[df['px_ask_0'] >= df['px_bid_0']]
            
            logger.info(f"  [OK] {filepath.name} cargado: {len(df):,} filas validas")
            return df
        
        except Exception as e:
            logger.error(f"  [ERROR] Error al cargar {filepath.name}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def load_sts_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Carga un archivo STS individual (trading status changes).
        
        Args:
            filepath: Path completo al archivo .csv.gz
            
        Returns:
            DataFrame con los cambios de estado, o None si hay error
        """
        try:
            # Intentar leer con diferentes encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(
                        filepath, 
                        compression='gzip',
                        sep=';',  # Separador: punto y coma
                        decimal='.',  # Decimal anglosajón (punto)
                        # NO especificar 'thousands' - los datos usan formato estándar
                        encoding=encoding,
                        low_memory=False
                    )
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error(f"  [ERROR] No se pudo leer {filepath.name}")
                return None
            
            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip()
            
            # Validar columnas críticas
            missing_cols = [col for col in self.STS_REQUIRED if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"  [ADVERTENCIA] Faltan columnas en {filepath.name}: {missing_cols}")
                return None
            
            # Convertir tipos
            df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
            df['epoch'] = (df['epoch'] * 1e9).astype('int64')
            df['market_trading_status'] = pd.to_numeric(df['market_trading_status'], errors='coerce').astype('Int64')
            
            # Eliminar NaNs
            df = df.dropna(subset=self.STS_REQUIRED)
            
            logger.info(f"  [OK] {filepath.name} cargado: {len(df):,} filas")
            return df
        
        except Exception as e:
            logger.error(f"  [ERROR] Error al cargar {filepath.name}: {str(e)}")
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
        
        # Buscar archivos
        qte_pattern = f"QTE_*_{isin}_*.csv.gz"
        sts_pattern = f"STS_*_{isin}_*.csv.gz"
        
        qte_files = glob.glob(str(self.data_dir / "**" / qte_pattern), recursive=True)
        sts_files = glob.glob(str(self.data_dir / "**" / sts_pattern), recursive=True)
        
        print(f"  Archivos QTE encontrados: {len(qte_files)}")
        print(f"  Archivos STS encontrados: {len(sts_files)}")
        
        # Inspección del primer archivo
        if len(qte_files) > 0:
            print(f"\n  Inspeccionando primer archivo...")
            self.inspect_file_structure(Path(qte_files[0]))
        
        # Organizar por venue (MIC)
        venue_data = {}
        
        for qte_file in qte_files:
            # Extraer MIC del filename
            parts = Path(qte_file).stem.split('_')
            mic = parts[4] if len(parts) >= 5 else None
            
            if mic is None:
                logger.warning(f"No se pudo extraer MIC de {qte_file}")
                continue
            
            print(f"\n  [PROCESANDO] Venue: {mic}")
            
            # Cargar QTE
            qte_df = self.load_qte_file(Path(qte_file))
            if qte_df is None:
                logger.warning(f"  [SKIP] {mic} - fallo al cargar QTE")
                continue
            
            # Buscar y cargar STS
            sts_file = qte_file.replace('QTE_', 'STS_')
            sts_df = None
            
            if Path(sts_file).exists():
                sts_df = self.load_sts_file(Path(sts_file))
                if sts_df is None:
                    logger.warning(f"  [ADVERTENCIA] STS no disponible para {mic}")
            else:
                logger.info(f"  [INFO] Archivo STS no encontrado para {mic}")
            
            # Almacenar
            venue_data[mic] = {
                'qte': qte_df,
                'sts': sts_df
            }
            
            print(f"    [OK] {mic}: {len(qte_df):,} snapshots cargados")
        
        if len(venue_data) == 0:
            logger.error("  [ERROR] No se pudo cargar ningun venue")
        else:
            print(f"\n[EXITO] Datos cargados para {len(venue_data)} venues: {list(venue_data.keys())}")
        
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