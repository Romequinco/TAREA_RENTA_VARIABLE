"""
================================================================================
data_cleaner_module.py - Módulo de Limpieza de Datos (Simplificado)
================================================================================

NOTA: Las funciones principales de limpieza están en data_loader_module.py:
- is_valid_price
- filter_valid_prices
- filter_continuous_trading

Este módulo mantiene compatibilidad con código existente que usa la clase DataCleaner.
Las funciones reales están en data_loader_module.py.
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from config_module import config
from data_loader_module import is_valid_price, filter_valid_prices, filter_continuous_trading

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clase wrapper para mantener compatibilidad con código existente.
    
    Las funciones reales de limpieza están en data_loader_module.py
    usando las funciones de data_loader_module.
    """
    
    def __init__(self):
        """Inicializa el limpiador."""
        self.magic_numbers = set(config.MAGIC_NUMBERS)
        self.valid_states = config.VALID_STATES.copy()
    
    @staticmethod
    def clean_magic_numbers(df: pd.DataFrame, price_columns: list = None) -> Tuple[pd.DataFrame, int]:
        """
        Elimina filas con magic numbers.
        
        Wrapper para mantener compatibilidad. La función real está en data_loader_module.
        """
        initial_len = len(df)
        if initial_len == 0:
            return df, 0
        
        # Usar filter_valid_prices que ya filtra magic numbers
        df_clean = filter_valid_prices(df)
        removed = initial_len - len(df_clean)
        
        return df_clean, removed
    
    def clean_all_venues(self, venue_data: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Aplica limpieza a todos los venues de un ISIN.
        
        Wrapper para mantener compatibilidad. La limpieza real se hace en
        load_data_for_isin() del data_loader_module.
        
        Args:
            venue_data: Dict {mic: {'qte': df, 'sts': df}}
            
        Returns:
            Dict {mic: cleaned_dataframe}
        """
        cleaned_data = {}
        
        for mic, data_dict in venue_data.items():
            try:
                qte_df = data_dict['qte']
                sts_df = data_dict.get('sts')
                
                # Aplicar filtros básicos
                qte_df = filter_valid_prices(qte_df)
                
                if sts_df is not None and not sts_df.empty:
                    # Mapear MIC a exchange name
                    mic_to_exchange = {
                        'XMAD': 'BME',
                        'AQXE': 'AQUIS',
                        'AQEU': 'AQUIS',
                        'CEUX': 'CBOE',
                        'TRQX': 'TURQUOISE',
                        'TQEX': 'TURQUOISE'
                    }
                    exchange = mic_to_exchange.get(mic, mic)
                    qte_df = filter_continuous_trading(qte_df, sts_df, exchange)
                
                if len(qte_df) > 0:
                    cleaned_data[mic] = qte_df
                else:
                    logger.warning(f"  {mic} excluded - no data after cleaning")
            
            except Exception as e:
                logger.error(f"  Error cleaning {mic}: {e}", exc_info=True)
        
        return cleaned_data
