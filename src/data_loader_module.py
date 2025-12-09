"""
================================================================================
data_loader_module.py - Módulo de Carga de Archivos QTE y STS
================================================================================

Funciones principales:
- load_qte_file: Carga archivos QTE (quotes)
- load_sts_file: Carga archivos STS (status)
- load_data_for_isin: Carga datos de un ISIN específico
- find_all_isins: Descubre todos los ISINs únicos en los datos

ESPECIFICACIONES:
- Formato de archivos: <type>_<session>_<isin>_<ticker>_<mic>_<part>.csv.gz
- Columnas QTE: epoch (int64), px_bid_0, px_ask_0, qty_bid_0, qty_ask_0
- Columnas STS: epoch (int64), market_trading_status
- Renombrado automático: px_bid_0 → bid, px_ask_0 → ask, etc.
================================================================================
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from config_module import config

logger = logging.getLogger(__name__)


def is_valid_price(price: float) -> bool:
    """
    Verifica si un precio es válido (no NaN, no magic number, y > 0).
    
    Args:
        price: Precio a validar
        
    Returns:
        True si el precio es válido, False en caso contrario
    """
    if pd.isna(price) or price <= 0:
        return False
    if price in config.MAGIC_NUMBERS:
        return False
    return True


def load_qte_file(file_path: Path) -> pd.DataFrame:
    """
    Carga un archivo QTE (quote).
    
    Proceso:
    1. Lee el archivo CSV comprimido con gzip
    2. Asegura que epoch sea int64
    3. Renombra columnas: px_bid_0 → bid, px_ask_0 → ask, etc.
    
    Args:
        file_path: Ruta al archivo QTE (.csv.gz)
        
    Returns:
        DataFrame con columnas renombradas: epoch, bid, ask, bidqty, askqty
    """
    try:
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep=';')
        
        # Asegurar que epoch sea int64 (crítico para precisión)
        if 'epoch' in df.columns:
            df['epoch'] = df['epoch'].astype('int64')
        
        # Renombrar columnas a formato estándar
        if 'px_bid_0' in df.columns:
            df = df.rename(columns={
                'px_bid_0': 'bid',
                'px_ask_0': 'ask',
                'qty_bid_0': 'bidqty',
                'qty_ask_0': 'askqty'
            })
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading QTE file {file_path}: {e}")
        return pd.DataFrame()


def load_sts_file(file_path: Path) -> pd.DataFrame:
    """
    Carga un archivo STS (status).
    
    Proceso:
    1. Lee el archivo CSV comprimido con gzip
    2. Asegura que epoch sea int64
    3. Mapea market_trading_status → status
    
    Args:
        file_path: Ruta al archivo STS (.csv.gz)
            
        Returns:
        DataFrame con columnas: epoch, status
    """
    try:
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep=';')
        
        # Asegurar que epoch sea int64
        if 'epoch' in df.columns:
            df['epoch'] = df['epoch'].astype('int64')
            
        # Mapear market_trading_status a status
        if 'market_trading_status' in df.columns:
            df['status'] = df['market_trading_status']
        elif 'trading_status' in df.columns:
            # Fallback para nombres alternativos
            df['status'] = df['trading_status']
        elif 'status' not in df.columns:
            # Si no existe ninguna, crear columna dummy
            df['status'] = None
        
        return df
            
    except Exception as e:
        logger.error(f"Error loading STS file {file_path}: {e}")
        return pd.DataFrame()


def filter_valid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra filas donde tanto bid como ask son precios válidos.
    
    Args:
        df: DataFrame con columnas 'bid' y 'ask'
        
    Returns:
        DataFrame filtrado con solo filas válidas
    """
    if df.empty:
        return df
    
    mask = df.apply(
        lambda row: is_valid_price(row.get('bid')) and is_valid_price(row.get('ask')),
        axis=1
    )
    return df[mask].copy()


def filter_continuous_trading(qte_df: pd.DataFrame, sts_df: pd.DataFrame, exchange: str) -> pd.DataFrame:
    """
    Filtra datos QTE para incluir solo períodos de continuous trading.
    
    Proceso:
    1. Hace merge_asof de QTE con STS usando direction='backward'
    2. Filtra por códigos de estado válidos para continuous trading
    
    Args:
        qte_df: DataFrame de quotes (con columnas: epoch, bid, ask, bidqty, askqty)
        sts_df: DataFrame de status (con columnas: epoch, status)
        exchange: Nombre del exchange ('BME', 'AQUIS', 'CBOE', 'TURQUOISE')
        
    Returns:
        DataFrame filtrado con solo continuous trading
    """
    if qte_df.empty or sts_df.empty:
        return qte_df
    
    # Obtener códigos de estado válidos para este exchange
    valid_statuses = config.CONTINUOUS_TRADING_STATUS.get(exchange, [])
    if not valid_statuses:
        return qte_df
    
    try:
        # Merge status information usando merge_asof (direction='backward')
        # Esto propaga el último estado conocido hacia adelante
        merged = pd.merge_asof(
            qte_df.sort_values('epoch'),
            sts_df[['epoch', 'status']].sort_values('epoch'),
            on='epoch',
            direction='backward'
        )
        
        # Verificar que la columna status existe y tiene datos
        if 'status' not in merged.columns:
            return pd.DataFrame()
        
        # Filtrar por estados válidos
        # Manejar tanto códigos numéricos como strings
        if pd.api.types.is_numeric_dtype(merged['status']):
            # Comparación numérica directa (más eficiente)
            mask = merged['status'].isin(valid_statuses)
        else:
            # Si status es string, convertir ambos a string para comparación
            merged['status_str'] = merged['status'].astype(str)
            valid_statuses_str = [str(s) for s in valid_statuses]
            mask = merged['status_str'].isin(valid_statuses_str)
            merged = merged.drop(columns=['status_str'])
        
        # Retornar datos filtrados (DataFrame vacío si no hay matches)
        result = merged[mask]
        
        return result
        
    except Exception as e:
        logger.warning(f"Error filtering continuous trading for {exchange}: {e}. Returning unfiltered data.")
        return qte_df


def load_data_for_isin(data_path: str, date: str, isin: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Carga y limpia datos QTE y STS para un ISIN específico en todos los exchanges.
    
    Proceso:
    1. Para cada exchange, busca archivos QTE y STS
    2. Carga y filtra precios válidos
    3. Filtra por continuous trading
    4. Retorna diccionario: {exchange: (qte_df, sts_df)}
        
        Args:
        data_path: Ruta al directorio de datos (ej: "DATA_BIG")
        date: Fecha en formato YYYY-MM-DD
            isin: International Securities Identification Number
            
        Returns:
            Diccionario con estructura:
            {
            'BME': (qte_df, sts_df),
            'AQUIS': (qte_df, sts_df),
                ...
            }
        """
    data_dict = {}
    
    for exchange in config.EXCHANGES:
        exchange_dir = Path(data_path) / f"{exchange}_{date}"
        
        if not exchange_dir.exists():
            continue
        
        # Buscar archivos QTE y STS (patrón flexible)
        qte_pattern = f"QTE_{date}_{isin}_*.csv.gz"
        sts_pattern = f"STS_{date}_{isin}_*.csv.gz"
        
        qte_files = list(exchange_dir.glob(qte_pattern))
        sts_files = list(exchange_dir.glob(sts_pattern))
        
        if not qte_files or not sts_files:
            continue
        
        # Cargar archivo QTE
        qte_df = load_qte_file(qte_files[0])
        qte_df = filter_valid_prices(qte_df)
        
        if qte_df.empty:
                continue
            
        # Cargar archivo STS
        sts_df = load_sts_file(sts_files[0])
        
        # Verificar que se cargó correctamente
        if sts_df is None or sts_df.empty:
                continue
            
        # Filtrar por continuous trading
        qte_df = filter_continuous_trading(qte_df, sts_df, exchange)
            
        if qte_df.empty:
                continue
            
        # Agregar identificador de exchange
        qte_df['exchange'] = exchange
        
        data_dict[exchange] = (qte_df, sts_df)
    
    return data_dict


# Alias para compatibilidad con código existente
load_data_for_isin_reference_format = load_data_for_isin


def find_all_isins(data_path: str, date: str) -> List[str]:
    """
    Descubre todos los ISINs únicos en los datos.
    
    Proceso:
    1. Busca todos los archivos QTE en todos los exchanges
    2. Extrae el ISIN del nombre del archivo
    3. Retorna lista ordenada de ISINs únicos
        
        Args:
        data_path: Ruta al directorio de datos
        date: Fecha en formato YYYY-MM-DD
            
        Returns:
        Lista ordenada de ISINs únicos
    """
    isins = set()
    
    for exchange in config.EXCHANGES:
        exchange_dir = Path(data_path) / f"{exchange}_{date}"
        if not exchange_dir.exists():
            continue
        
        # Buscar todos los archivos QTE
        qte_files = list(exchange_dir.glob(f"QTE_{date}_*.csv.gz"))
        
        for file_path in qte_files:
            # Extraer ISIN del nombre del archivo
            parts = file_path.stem.replace('.csv', '').split('_')
            if len(parts) >= 3:
                isin = parts[2]
                isins.add(isin)
    
    return sorted(list(isins))


# ============================================================================
# Clase DataLoader (compatibilidad con código existente)
# ============================================================================

class DataLoader:
    """
    Clase wrapper para mantener compatibilidad con código existente.
    
    Internamente usa las funciones del módulo.
    """
    
    def __init__(self, data_path: str):
        """
        Inicializa el cargador de datos.
        
        Args:
            data_path: Ruta al directorio de datos
        """
        self.data_path = data_path
        logger.info(f"DataLoader initialized with data_path: {data_path}")
    
    def discover_isins(self, date: str = None) -> List[str]:
        """
        Descubre todos los ISINs únicos en los datos.
        
        Wrapper para find_all_isins().
        
        Args:
            date: Fecha en formato YYYY-MM-DD. Si es None, intenta detectarla.
            
        Returns:
            Lista ordenada de ISINs únicos
        """
        # Si no se proporciona fecha, intentar detectarla
        if date is None:
            # Buscar la primera fecha disponible en el directorio
            data_dir = Path(self.data_path)
            if not data_dir.exists():
                raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")
            
            # Buscar directorios con formato {exchange}_{date}
            date_candidates = set()
            for item in data_dir.iterdir():
                if item.is_dir():
                    parts = item.name.split('_')
                    if len(parts) >= 2:
                        date_candidates.add('_'.join(parts[1:]))
            
            if not date_candidates:
                raise ValueError("No se pudo detectar la fecha. Proporciona el parámetro 'date'.")
            
            # Usar la primera fecha encontrada
            date = sorted(date_candidates)[0]
            logger.info(f"Fecha detectada automáticamente: {date}")
        
        return find_all_isins(self.data_path, date)
    
    def load_isin_data(self, isin: str, date: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Carga datos de un ISIN específico.
        
        Wrapper para load_data_for_isin() que convierte el formato de salida
        a Dict[str, Dict[str, DataFrame]] para compatibilidad.
        
        Args:
            isin: ISIN a cargar
            date: Fecha en formato YYYY-MM-DD. Si es None, intenta detectarla.
            
        Returns:
            Dict con estructura: {mic: {'qte': df, 'sts': df}}
        """
        # Si no se proporciona fecha, intentar detectarla
        if date is None:
            # Buscar la primera fecha disponible en el directorio
            data_dir = Path(self.data_path)
            if not data_dir.exists():
                raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")
            
            # Buscar directorios con formato {exchange}_{date}
            date_candidates = set()
            for item in data_dir.iterdir():
                if item.is_dir():
                    parts = item.name.split('_')
                    if len(parts) >= 2:
                        date_candidates.add('_'.join(parts[1:]))
            
            if not date_candidates:
                raise ValueError("No se pudo detectar la fecha. Proporciona el parámetro 'date'.")
            
            # Usar la primera fecha encontrada
            date = sorted(date_candidates)[0]
            logger.info(f"Fecha detectada automáticamente: {date}")
        
        # Cargar datos usando la función del módulo
        data_dict = load_data_for_isin(self.data_path, date, isin)
        
        # Convertir formato: Dict[str, Tuple[DataFrame, DataFrame]] → Dict[str, Dict[str, DataFrame]]
        result = {}
        for exchange, (qte_df, sts_df) in data_dict.items():
            # Mapear exchange a MIC para compatibilidad
            exchange_to_mic = {
                'BME': 'XMAD',
                'AQUIS': 'AQXE',
                'CBOE': 'CEUX',
                'TURQUOISE': 'TRQX'
            }
            mic = exchange_to_mic.get(exchange, exchange)
            result[mic] = {
                'qte': qte_df,
                'sts': sts_df
            }
        
        return result
