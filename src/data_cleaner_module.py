"""
================================================================================
data_cleaner.py - Módulo de Limpieza y Validación de Datos
================================================================================
Responsabilidades:
- Filtrado de magic numbers (precios inválidos del vendor)
- Validación de precios y cantidades
- Detección de crossed books (bid >= ask)
- Filtrado por market status (solo continuous trading)

Pipeline de limpieza (FASE 2):
1. Clean magic numbers
2. Clean invalid prices
3. Clean crossed books
4. Filter by market status
================================================================================
"""

import pandas as pd
import logging
from typing import Dict

from config_module import config

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clase responsable de aplicar filtros de calidad de datos.
    
    CRÍTICO: La limpieza es esencial para evitar señales de arbitraje falsas.
    Los magic numbers y estados de mercado inválidos generarían oportunidades
    que en realidad no son ejecutables.
    """
    
    @staticmethod
    def clean_magic_numbers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina snapshots con magic numbers en precios.
        
        CRÍTICO: Los magic numbers NO son precios reales. Son códigos especiales
        del vendor para indicar estados como "Market Order", "Pegged Order", etc.
        
        Si no se filtran, se detectarían oportunidades falsas con profits de
        millones de euros.
        
        Fuente: Arbitrage study in BME.docx - Section 2.A "Magic Numbers"
        
        Args:
            df: DataFrame con columnas px_bid_0 y px_ask_0
            
        Returns:
            DataFrame filtrado (sin magic numbers)
        """
        initial_len = len(df)
        
        # Crear máscaras: True si NO es magic number
        bid_mask = ~df['px_bid_0'].isin(config.MAGIC_NUMBERS)
        ask_mask = ~df['px_ask_0'].isin(config.MAGIC_NUMBERS)
        
        # Aplicar filtro combinado
        df_clean = df[bid_mask & ask_mask].copy()
        
        removed = initial_len - len(df_clean)
        if removed > 0:
            pct = removed / initial_len * 100
            logger.info(f"    Removed {removed:,} magic numbers ({pct:.2f}%)")
        
        return df_clean
    
    @staticmethod
    def clean_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida que todos los precios y cantidades sean positivos y no-NaN.
        
        Validaciones:
        - px_bid_0 > 0
        - px_ask_0 > 0
        - qty_bid_0 > 0
        - qty_ask_0 > 0
        - No NaN values
        
        Args:
            df: DataFrame con columnas de precios y cantidades
            
        Returns:
            DataFrame con precios válidos
        """
        initial_len = len(df)
        
        # Máscara combinada de validaciones
        mask = (
            (df['px_bid_0'] > 0) &
            (df['px_ask_0'] > 0) &
            (df['qty_bid_0'] > 0) &
            (df['qty_ask_0'] > 0) &
            (df['px_bid_0'].notna()) &
            (df['px_ask_0'].notna())
        )
        
        df_clean = df[mask].copy()
        
        removed = initial_len - len(df_clean)
        if removed > 0:
            logger.info(f"    Removed {removed:,} invalid prices")
        
        return df_clean
    
    @staticmethod
    def clean_crossed_book(df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina snapshots donde bid >= ask dentro del mismo venue.
        
        Un "crossed book" indica error en los datos. En un mercado normal,
        el mejor bid siempre debe ser menor que el mejor ask dentro del
        mismo venue. Si bid >= ask, habría arbitraje instantáneo y el
        mercado se auto-corregiría inmediatamente.
        
        NOTA: Cross-venue arbitrage (bid de venue A > ask de venue B) es
        lo que QUEREMOS detectar. Esto solo filtra anomalías internas.
        
        Args:
            df: DataFrame con px_bid_0 y px_ask_0
            
        Returns:
            DataFrame sin crossed books
        """
        initial_len = len(df)
        
        # Condición normal: bid < ask
        mask = df['px_bid_0'] < df['px_ask_0']
        df_clean = df[mask].copy()
        
        removed = initial_len - len(df_clean)
        if removed > 0:
            logger.warning(f"    Removed {removed:,} crossed books (bid >= ask)")
        
        return df_clean
    
    @staticmethod
    def filter_by_market_status(qte_df: pd.DataFrame, 
                                sts_df: pd.DataFrame, 
                                mic: str) -> pd.DataFrame:
        """
        Filtra snapshots para mantener SOLO los que ocurren durante
        continuous trading.
        
        CRÍTICO: Operar durante auctions, halts o pre-open generaría señales
        falsas. Las órdenes no se ejecutarían instantáneamente en esos estados.
        
        Método:
        - Usa pd.merge_asof con direction='backward' para propagar el último
          estado conocido hacia adelante
        - Filtra por códigos de continuous trading según venue
        - Si no hay códigos válidos conocidos, mantiene los datos con advertencia
        
        Fuente: Arbitrage study in BME.docx - Section 2.B "Market Status Codes"
        
        Args:
            qte_df: DataFrame con quotes
            sts_df: DataFrame con trading status
            mic: Market Identifier Code (XMAD, AQXE, CEUX, TRQX)
            
        Returns:
            DataFrame filtrado (solo continuous trading si es posible)
        """
        # Validar datos de entrada
        if sts_df is None or len(sts_df) == 0:
            logger.warning(f"    No STS data for {mic}, skipping status filter")
            return qte_df
        
        if mic not in config.VALID_STATES:
            logger.warning(f"    Unknown MIC {mic}, skipping status filter")
            return qte_df
        
        initial_len = len(qte_df)
        
        # Verificar si la columna existe
        if 'market_trading_status' not in sts_df.columns:
            logger.warning(f"    Column 'market_trading_status' not found in STS for {mic}")
            return qte_df
        
        # CORRECCIÓN: Verificar si los códigos válidos existen en los datos
        valid_codes = config.VALID_STATES[mic]
        actual_codes = set(sts_df['market_trading_status'].dropna().unique())
        matching_codes = set(valid_codes).intersection(actual_codes)
        
        if len(matching_codes) == 0:
            logger.warning(f"    No matching trading status codes found for {mic}")
            logger.warning(f"    Expected codes: {valid_codes}")
            logger.warning(f"    Found codes: {sorted(actual_codes)}")
            logger.warning(f"    Keeping all data without status filtering")
            return qte_df
        
        # Ordenar ambos DataFrames por timestamp
        qte_sorted = qte_df.sort_values('epoch').copy()
        sts_sorted = sts_df[['epoch', 'market_trading_status']].sort_values('epoch').copy()
        
        # Merge asof: Asigna a cada snapshot el estado más reciente anterior
        # direction='backward' significa "usar el último valor conocido"
        try:
            merged = pd.merge_asof(
                qte_sorted,
                sts_sorted,
                on='epoch',
                direction='backward'
            )
        except Exception as e:
            logger.error(f"    Error in merge_asof for {mic}: {e}")
            return qte_df
        
        # Filtrar por códigos válidos de continuous trading
        merged_filtered = merged[
            merged['market_trading_status'].isin(matching_codes)
        ].copy()
        
        # Limpiar columna temporal
        if 'market_trading_status' in merged_filtered.columns:
            merged_filtered = merged_filtered.drop('market_trading_status', axis=1)
        
        removed = initial_len - len(merged_filtered)
        
        # CORRECCIÓN: Solo aplicar filtro si no elimina TODOS los datos
        if len(merged_filtered) == 0:
            logger.warning(f"    Status filtering would remove ALL data for {mic}")
            logger.warning(f"    Keeping original data without status filtering")
            return qte_df
        
        if removed > 0:
            pct = removed / initial_len * 100
            logger.info(f"    Removed {removed:,} non-trading snapshots ({pct:.2f}%)")
        
        return merged_filtered
    
    def clean_venue_data(self, venue_dict: Dict, mic: str) -> pd.DataFrame:
        """
        Aplica el pipeline completo de limpieza a un venue.
        
        Pipeline secuencial:
        1. Magic numbers
        2. Invalid prices
        3. Crossed books
        4. Market status (con filtrado adaptativo)
        
        Args:
            venue_dict: Dict con keys 'qte' y 'sts'
            mic: Market Identifier Code
            
        Returns:
            DataFrame limpio y validado
        """
        print(f"\n  [LIMPIEZA] {mic}...")
        
        qte_df = venue_dict['qte']
        sts_df = venue_dict['sts']
        
        print(f"    Snapshots iniciales: {len(qte_df):,}")
        
        # Aplicar filtros secuencialmente
        df = qte_df.copy()
        df = self.clean_magic_numbers(df)
        df = self.clean_invalid_prices(df)
        df = self.clean_crossed_book(df)
        df = self.filter_by_market_status(df, sts_df, mic)
        
        print(f"    [OK] Snapshots finales: {len(df):,} (limpios)")
        
        # Advertencia si queda muy poco
        if len(df) == 0:
            logger.warning(f"  {mic} has 0 snapshots after cleaning")
        elif len(df) < len(qte_df) * 0.01:  # Menos del 1%
            pct = (len(df) / len(qte_df)) * 100
            logger.warning(f"  {mic} retained only {pct:.2f}% of original data")
        
        return df
    
    def clean_all_venues(self, venue_data: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Aplica limpieza a todos los venues de un ISIN.
        
        Args:
            venue_data: Dict {mic: {'qte': df, 'sts': df}}
            
        Returns:
            Dict {mic: cleaned_dataframe}
        """
        print("\n" + "=" * 80)
        print("LIMPIEZA Y VALIDACIÓN DE DATOS")
        print("=" * 80)
        
        cleaned_data = {}
        
        for mic, data_dict in venue_data.items():
            try:
                cleaned_df = self.clean_venue_data(data_dict, mic)
                
                if len(cleaned_df) > 0:
                    cleaned_data[mic] = cleaned_df
                else:
                    logger.warning(f"  {mic} excluded - no data after cleaning")
            
            except Exception as e:
                logger.error(f"  Error cleaning {mic}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        print(f"\n[EXITO] Limpieza completada para {len(cleaned_data)} venues")
        
        if len(cleaned_data) == 0:
            logger.error("  [CRITICO] No venues survived cleaning!")
        
        return cleaned_data