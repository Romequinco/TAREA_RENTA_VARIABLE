"""
================================================================================
data_loader.py - Módulo de Carga de Archivos QTE y STS
================================================================================
Responsabilidades:
- Descubrimiento de ISINs disponibles
- Carga de archivos .csv.gz (QTE y STS)
- Organización de datos por venue (MIC)
- Validación de columnas requeridas
================================================================================
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Clase responsable de la ingesta de datos desde archivos comprimidos.
    
    Archivos gestionados:
    - QTE (Quotes): Snapshots del order book (bid/ask prices & quantities)
    - STS (Status): Cambios de estado del mercado (trading status codes)
    
    Formato de archivos:
    - Naming: <type>_<session>_<isin>_<ticker>_<mic>_<part>.csv.gz
    - Book Identity: (session, isin, mic, ticker)
    """
    
    def __init__(self, data_dir: Path):
        """
        Inicializa el loader con el directorio de datos.
        
        Args:
            data_dir: Path al directorio conteniendo subcarpetas con archivos .csv.gz
        """
        self.data_dir = data_dir
        logger.info(f"DataLoader initialized: {data_dir}")
    
    def discover_isins(self) -> List[str]:
        """
        Descubre todos los ISINs únicos disponibles en el directorio.
        
        Proceso:
        1. Busca recursivamente todos los archivos QTE_*.csv.gz
        2. Extrae el ISIN del filename (posición 2 después de split por '_')
        3. Retorna lista ordenada de ISINs únicos
        
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
        
        print(f"✓ Encontrados {len(isins_list)} ISINs únicos")
        if len(isins_list) > 0:
            print(f"  Primeros 5: {isins_list[:5]}")
        
        return isins_list
    
    def load_qte_file(self, filepath: Path) -> pd.DataFrame:
        """
        Carga un archivo QTE individual (order book snapshots).
        
        Columnas requeridas:
        - epoch: Timestamp en microsegundos UTC
        - px_bid_0, px_ask_0: Best bid/ask prices
        - qty_bid_0, qty_ask_0: Quantities at best prices
        
        Args:
            filepath: Path completo al archivo .csv.gz
            
        Returns:
            DataFrame con los snapshots, o None si hay error
        """
        try:
            # Leer archivo comprimido
            df = pd.read_csv(filepath, compression='gzip')
            
            # Validar columnas críticas
            required_cols = ['epoch', 'px_bid_0', 'px_ask_0', 'qty_bid_0', 'qty_ask_0']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in {filepath.name}: {missing_cols}")
                return None
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {e}")
            return None
    
    def load_sts_file(self, filepath: Path) -> pd.DataFrame:
        """
        Carga un archivo STS individual (trading status changes).
        
        Columnas requeridas:
        - epoch: Timestamp en microsegundos UTC
        - market_trading_status: Código de estado del mercado
        
        Args:
            filepath: Path completo al archivo .csv.gz
            
        Returns:
            DataFrame con los cambios de estado, o None si hay error
        """
        try:
            # Leer archivo comprimido
            df = pd.read_csv(filepath, compression='gzip')
            
            # Validar columnas críticas
            required_cols = ['epoch', 'market_trading_status']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in {filepath.name}: {missing_cols}")
                return None
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {e}")
            return None
    
    def load_isin_data(self, isin: str) -> Dict[str, Dict]:
        """
        Carga TODOS los datos (QTE + STS) para un ISIN específico.
        Organiza los datos por venue (Market Identifier Code).
        
        Proceso:
        1. Buscar todos los archivos QTE y STS para el ISIN
        2. Para cada archivo QTE, extraer el MIC del filename
        3. Cargar el QTE y buscar el STS correspondiente
        4. Organizar en diccionario por venue
        
        Args:
            isin: International Securities Identification Number
            
        Returns:
            Diccionario con estructura:
            {
                'XMAD': {'qte': DataFrame, 'sts': DataFrame},
                'AQXE': {'qte': DataFrame, 'sts': DataFrame},
                'CEUX': {'qte': DataFrame, 'sts': DataFrame},
                'TRQX': {'qte': DataFrame, 'sts': DataFrame}
            }
        """
        print("\n" + "=" * 80)
        print(f"CARGANDO DATOS PARA ISIN: {isin}")
        print("=" * 80)
        
        # Definir patrones de búsqueda
        qte_pattern = f"QTE_*_{isin}_*.csv.gz"
        sts_pattern = f"STS_*_{isin}_*.csv.gz"
        
        # Buscar archivos
        qte_files = glob.glob(
            str(self.data_dir / "**" / qte_pattern), 
            recursive=True
        )
        sts_files = glob.glob(
            str(self.data_dir / "**" / sts_pattern), 
            recursive=True
        )
        
        print(f"  Archivos QTE encontrados: {len(qte_files)}")
        print(f"  Archivos STS encontrados: {len(sts_files)}")
        
        # Organizar por venue (MIC)
        venue_data = {}
        
        for qte_file in qte_files:
            # Extraer MIC del filename
            # Formato: QTE_session_isin_ticker_mic_part.csv.gz
            parts = Path(qte_file).stem.split('_')
            mic = parts[4] if len(parts) >= 5 else None
            
            if mic is None:
                logger.warning(f"Cannot extract MIC from {qte_file}")
                continue
            
            # Cargar archivo QTE
            qte_df = self.load_qte_file(Path(qte_file))
            if qte_df is None:
                continue
            
            # Buscar archivo STS correspondiente
            sts_file = qte_file.replace('QTE_', 'STS_')
            sts_df = None
            
            if Path(sts_file).exists():
                sts_df = self.load_sts_file(Path(sts_file))
            else:
                logger.warning(f"STS file not found for {mic}")
            
            # Almacenar en diccionario
            venue_data[mic] = {
                'qte': qte_df,
                'sts': sts_df
            }
            
            print(f"  ✓ {mic}: {len(qte_df):,} snapshots cargados")
        
        print(f"\n✓ Datos cargados para {len(venue_data)} venues: {list(venue_data.keys())}")
        
        return venue_data
    
    def get_book_identity(self, filepath: Path) -> tuple:
        """
        Extrae el Book Identity Key del filename.
        
        Book Identity = (session, isin, mic, ticker)
        Este tuple identifica unívocamente cada order book.
        
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
