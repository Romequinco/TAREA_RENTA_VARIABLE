"""
================================================================================
consolidator_module.py - Módulo de Consolidated Tape
================================================================================

Funciones principales:
- create_consolidated_tape: Crea el tape consolidado fusionando datos de todos los exchanges
- get_quote_at_epoch: Consulta eficiente del tape a un epoch específico

ALGORITMO:
1. Renombrar columnas por exchange: bid → {exchange}_bid, etc.
2. Merge iterativo con outer join en 'epoch'
3. Ordenar por epoch
4. Forward fill para cada exchange (CRÍTICO: asume que el último precio conocido
   sigue vigente hasta el próximo update)
5. Establecer epoch como índice (pero mantenerlo como columna también)

ESTRUCTURA DEL TAPE:
| epoch | BME_bid | BME_ask | BME_bidqty | BME_askqty |
|       | AQUIS_bid | AQUIS_ask | AQUIS_bidqty | AQUIS_askqty |
|       | CBOE_bid | CBOE_ask | CBOE_bidqty | CBOE_askqty |
|       | TURQUOISE_bid | TURQUOISE_ask | TURQUOISE_bidqty | TURQUOISE_askqty |
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from config_module import config

logger = logging.getLogger(__name__)


def create_consolidated_tape(data_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """
    Fusiona quotes de todos los exchanges en un único consolidated tape.
    
    Proceso:
    1. Para cada exchange, selecciona columnas relevantes (epoch, bid, ask, bidqty, askqty)
    2. Renombra columnas con prefijo del exchange
    3. Hace merge iterativo con outer join
    4. Ordena por epoch
    5. Aplica forward fill por exchange
    6. Establece epoch como índice
    
    Args:
        data_dict: Diccionario con estructura {exchange: (qte_df, sts_df)}
                   También acepta Dict[str, DataFrame] para compatibilidad
        
    Returns:
        DataFrame consolidado con epoch como índice y columnas:
        {exchange}_bid, {exchange}_ask, {exchange}_bidqty, {exchange}_askqty
    """
    if not data_dict:
        return pd.DataFrame()
    
    all_quotes = []
    
    # Verificar formato y convertir si es necesario
    for exchange, value in data_dict.items():
        # Si es una tupla (formato correcto)
        if isinstance(value, tuple) and len(value) == 2:
            qte_df, _ = value
        # Si es solo un DataFrame (formato antiguo - compatibilidad)
        elif isinstance(value, pd.DataFrame):
            qte_df = value
        else:
            logger.warning(f"Formato inesperado para {exchange}, saltando...")
            continue
        if qte_df.empty:
            continue
        
        # Seleccionar columnas relevantes
        cols = ['epoch', 'bid', 'ask', 'bidqty', 'askqty']
        quote_df = qte_df[cols].copy()
        
        # Agregar prefijo del exchange a columnas bid/ask
        quote_df = quote_df.rename(columns={
            'bid': f'{exchange}_bid',
            'ask': f'{exchange}_ask',
            'bidqty': f'{exchange}_bidqty',
            'askqty': f'{exchange}_askqty'
        })
        
        all_quotes.append(quote_df)
    
    if not all_quotes:
        return pd.DataFrame()
    
    # Merge iterativo: empezar con el primer exchange y hacer outer merge con los demás
    consolidated = all_quotes[0]
    for quote_df in all_quotes[1:]:
        consolidated = pd.merge(consolidated, quote_df, on='epoch', how='outer')
    
    # Ordenar por epoch
    consolidated = consolidated.sort_values('epoch').reset_index(drop=True)
    
    # Aplicar forward fill para cada exchange (CRÍTICO)
    # Asunción de market microstructure: el último precio conocido sigue vigente
    # hasta que llegue un nuevo update. Esto es estándar en análisis de order books.
    for exchange in data_dict.keys():
        for col in [f'{exchange}_bid', f'{exchange}_ask', f'{exchange}_bidqty', f'{exchange}_askqty']:
            if col in consolidated.columns:
                consolidated[col] = consolidated[col].ffill()
    
    # Establecer epoch como índice (pero mantenerlo como columna también)
    consolidated = consolidated.set_index('epoch')
    consolidated['epoch'] = consolidated.index
    
    return consolidated


def get_quote_at_epoch(consolidated: pd.DataFrame, target_epoch: int, method: str = 'nearest') -> Optional[pd.Series]:
    """
    Consulta eficiente del consolidated tape en un epoch específico.
    
    Usa searchsorted para búsqueda binaria eficiente (O(log n)).
    
    Args:
        consolidated: DataFrame consolidado con epoch como índice
        target_epoch: Epoch objetivo en microsegundos
        method: Método de búsqueda ('nearest' = más cercano)
        
    Returns:
        Series con los datos del quote en el epoch más cercano, o None si no hay datos
    """
    if consolidated.empty:
        return None
    
    if method == 'nearest':
        # Usar searchsorted para búsqueda binaria eficiente
        idx = np.searchsorted(consolidated.index, target_epoch, side='left')
        
        # Manejar casos límite
        if idx == 0:
            return consolidated.iloc[0]
        elif idx >= len(consolidated):
            return consolidated.iloc[-1]
        else:
            # Elegir el más cercano
            left_epoch = consolidated.index[idx - 1]
            right_epoch = consolidated.index[idx]
            
            if abs(target_epoch - left_epoch) <= abs(target_epoch - right_epoch):
                return consolidated.iloc[idx - 1]
            else:
                return consolidated.iloc[idx]
    
    return None


# ============================================================================
# Clase ConsolidatedTape (compatibilidad con código existente)
# ============================================================================

class ConsolidatedTape:
    """
    Clase wrapper para mantener compatibilidad con código existente.
    
    Internamente usa las funciones del módulo.
    """
    
    def __init__(self, time_bin_ms: Optional[int] = None):
        """
        Args:
            time_bin_ms: Ventana de redondeo temporal (no usado)
        """
        self.time_bin_ms = time_bin_ms
        logger.info(f"ConsolidatedTape initialized (time_bin={time_bin_ms}ms)" if time_bin_ms else "ConsolidatedTape initialized")
    
    def create_tape(self, venue_data: Dict) -> Optional[pd.DataFrame]:
        """
        Crea el consolidated tape.
        
        Args:
            venue_data: Puede ser Dict[str, DataFrame] o Dict[str, Tuple[DataFrame, DataFrame]]
        
        Returns:
            DataFrame consolidado
        """
        # Convertir formato si es necesario
        if venue_data and isinstance(next(iter(venue_data.values())), tuple):
            # Formato: Dict[str, Tuple[DataFrame, DataFrame]]
            data_dict = venue_data
        else:
            # Formato: Dict[str, DataFrame] - convertir a formato esperado
            data_dict = {k: (v, pd.DataFrame()) for k, v in venue_data.items()}
        
        return create_consolidated_tape(data_dict)
    
    @staticmethod
    def validate_tape(df: pd.DataFrame) -> bool:
        """
        Valida el consolidated tape.
        
        Args:
            df: DataFrame consolidado
            
        Returns:
            True si es válido
        """
        if df is None or len(df) == 0:
            return False
        
        # Validaciones básicas
        if 'epoch' not in df.columns:
            return False
        
        # Verificar que hay al menos una columna de precio
        price_cols = [col for col in df.columns if '_bid' in col or '_ask' in col]
        if not price_cols:
            return False
        
        return True
