"""
================================================================================
consolidator.py - Módulo de Consolidated Tape
================================================================================
Responsabilidades:
- Merge de datos multi-venue en un único DataFrame
- Forward fill para propagar últimos precios conocidos
- Validación del tape consolidado

Estructura del Consolidated Tape:
    epoch | XMAD_bid | XMAD_ask | XMAD_bid_qty | XMAD_ask_qty |
          | AQXE_bid | AQXE_ask | AQXE_bid_qty | AQXE_ask_qty |
          | CEUX_bid | CEUX_ask | CEUX_bid_qty | CEUX_ask_qty |
          | TRQX_bid | TRQX_ask | TRQX_bid_qty | TRQX_ask_qty |
================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

from config_module import config

logger = logging.getLogger(__name__)


class ConsolidatedTape:
    """
    Crea y valida el Consolidated Tape multi-venue.
    
    El tape consolidado permite comparar precios de todos los venues en
    el mismo instante temporal, que es esencial para detectar arbitraje.
    """
    
    @staticmethod
    def create_tape(venue_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge de todos los venues en un único DataFrame.
        
        Proceso:
        1. Para cada venue, renombrar columnas con prefijo del MIC
        2. Outer join iterativo por 'epoch' (timestamp)
        3. Ordenar por timestamp
        4. Forward fill para propagar últimos precios conocidos
        5. Eliminar filas iniciales con NaNs
        
        Forward Fill Rationale:
        En market microstructure, se asume que el último precio conocido
        sigue vigente hasta que llegue un nuevo update. Esto es estándar
        y refleja la realidad del mercado.
        
        Fuente: arbitrage_architecture.md - Sección 3 FASE 3
        
        Args:
            venue_data: Dict {mic: DataFrame con [epoch, px_bid_0, px_ask_0, qty_bid_0, qty_ask_0]}
            
        Returns:
            DataFrame consolidado con todas las columnas de todos los venues
        """
        print("\n" + "=" * 80)
        print("CREANDO CONSOLIDATED TAPE")
        print("=" * 80)
        
        if len(venue_data) == 0:
            logger.error("  No venue data to consolidate")
            return None
        
        print(f"  Venues a consolidar: {list(venue_data.keys())}")
        
        # ====================================================================
        # PASO 1: Preparar DataFrames individuales con renombrado
        # ====================================================================
        dfs = []
        
        for mic, df in venue_data.items():
            # Seleccionar columnas relevantes
            df_subset = df[['epoch', 'px_bid_0', 'px_ask_0', 
                           'qty_bid_0', 'qty_ask_0']].copy()
            
            # Renombrar con prefijo del venue
            df_subset.columns = [
                'epoch',
                f'{mic}_bid',
                f'{mic}_ask',
                f'{mic}_bid_qty',
                f'{mic}_ask_qty'
            ]
            
            dfs.append(df_subset)
            print(f"   {mic}: {len(df_subset):,} snapshots preparados")
        
        # ====================================================================
        # PASO 2: Merge iterativo con outer join
        # ====================================================================
        print("\n  Ejecutando outer merge...")
        
        consolidated = dfs[0]
        
        for i, df in enumerate(dfs[1:], 1):
            consolidated = pd.merge(
                consolidated,
                df,
                on='epoch',
                how='outer'  # Mantener todos los timestamps de todos los venues
            )
            print(f"    Merge {i}/{len(dfs)-1} completado")
        
        # ====================================================================
        # PASO 3: Ordenar por timestamp
        # ====================================================================
        consolidated = consolidated.sort_values('epoch').reset_index(drop=True)
        
        print(f"\n  ✓ Tape consolidado creado: {consolidated.shape}")
        print(f"    - Timestamps únicos: {len(consolidated):,}")
        print(f"    - Columnas totales: {len(consolidated.columns)}")
        print(f"    - Rango temporal: {consolidated['epoch'].min()} to {consolidated['epoch'].max()}")
        
        # ====================================================================
        # PASO 4: Forward fill para propagar últimos precios
        # ====================================================================
        print("\n  Aplicando forward fill...")
        
        nans_before = consolidated.isna().sum().sum()
        print(f"    NaNs antes: {nans_before:,}")
        
        # Forward fill: Propagar último valor conocido hacia adelante
        consolidated = consolidated.fillna(method='ffill')
        
        nans_after = consolidated.isna().sum().sum()
        print(f"    NaNs después: {nans_after:,}")
        
        # ====================================================================
        # PASO 5: Eliminar primeras filas con NaNs
        # ====================================================================
        # Las primeras filas pueden tener NaNs si un venue no tenía datos
        # al inicio. Eliminarlas para tener un tape completamente válido.
        
        initial_nans = consolidated.isna().sum(axis=1)
        
        if initial_nans.max() > 0:
            first_complete_row = (initial_nans == 0).idxmax()
            
            if first_complete_row > 0:
                print(f"    Eliminando primeras {first_complete_row} filas incompletas")
                consolidated = consolidated.iloc[first_complete_row:].reset_index(drop=True)
        
        print(f"\n  Tape final: {consolidated.shape}")
        
        return consolidated
    
    @staticmethod
    def validate_tape(df: pd.DataFrame) -> bool:
        """
        Validaciones críticas del consolidated tape.
        
        Validaciones (Fuente: arbitrage_architecture.md - Sección 4.1):
        1. No NaNs en el tape final
        2. Timestamps monotónicamente crecientes
        3. No negative spreads dentro de cada venue
        4. No precios excesivamente altos (residual magic numbers)
        
        Args:
            df: Consolidated tape
            
        Returns:
            True si todas las validaciones pasan
        """
        print("\n" + "=" * 80)
        print("VALIDANDO CONSOLIDATED TAPE")
        print("=" * 80)
        
        # ====================================================================
        # Validación 1: No NaNs
        # ====================================================================
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.error(f"  Found {nan_count:,} NaNs in consolidated tape")
            return False
        print("  ✓ No NaNs encontrados")
        
        # ====================================================================
        # Validación 2: Timestamps monotónicos
        # ====================================================================
        if not df['epoch'].is_monotonic_increasing:
            logger.error(" Timestamps are not monotonically increasing")
            return False
        print("  ✓ Timestamps monotónicamente crecientes")
        
        # ====================================================================
        # Validación 3: No negative spreads dentro de cada venue
        # ====================================================================
        bid_cols = [col for col in df.columns if col.endswith('_bid')]
        
        all_spreads_valid = True
        for bid_col in bid_cols:
            ask_col = bid_col.replace('_bid', '_ask')
            
            if ask_col in df.columns:
                spread = df[ask_col] - df[bid_col]
                
                if (spread < 0).any():
                    venue = bid_col.replace('_bid', '')
                    logger.error(f"  Negative spread found in {venue}")
                    all_spreads_valid = False
        
        if all_spreads_valid:
            print("  No negative spreads dentro de venues")
        else:
            return False
        
        # ====================================================================
        # Validación 4: No precios excesivos
        # ====================================================================
        price_cols = [col for col in df.columns if 'bid' in col or 'ask' in col]
        max_price = df[price_cols].max().max()
        
        if max_price > config.MAX_REASONABLE_PRICE:
            logger.warning(f"  Precio sospechosamente alto: €{max_price:.2f}")
        else:
            print(f"  Precios razonables (max: €{max_price:.2f})")
        
        print("\n  VALIDACIÓN EXITOSA")
        return True
    
    @staticmethod
    def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estadísticas descriptivas del consolidated tape.
        
        Args:
            df: Consolidated tape
            
        Returns:
            DataFrame con estadísticas por venue
        """
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS DEL CONSOLIDATED TAPE")
        print("=" * 80)
        
        # Extraer venues
        bid_cols = [col for col in df.columns if col.endswith('_bid')]
        venues = [col.replace('_bid', '') for col in bid_cols]
        
        stats_list = []
        
        for venue in venues:
            bid_col = f'{venue}_bid'
            ask_col = f'{venue}_ask'
            
            if bid_col in df.columns and ask_col in df.columns:
                spread = df[ask_col] - df[bid_col]
                mid_price = (df[bid_col] + df[ask_col]) / 2
                
                stats = {
                    'Venue': venue,
                    'Avg Bid (€)': df[bid_col].mean(),
                    'Avg Ask (€)': df[ask_col].mean(),
                    'Avg Mid (€)': mid_price.mean(),
                    'Avg Spread (€)': spread.mean(),
                    'Min Spread (€)': spread.min(),
                    'Max Spread (€)': spread.max(),
                    'Spread Std (€)': spread.std()
                }
                
                stats_list.append(stats)
                
                print(f"\n{venue}:")
                print(f"  Bid medio: €{stats['Avg Bid (€)']:.4f}")
                print(f"  Ask medio: €{stats['Avg Ask (€)']:.4f}")
                print(f"  Spread medio: €{stats['Avg Spread (€)']:.4f}")
                print(f"  Spread min/max: €{stats['Min Spread (€)']:.4f} / €{stats['Max Spread (€)']:.4f}")
        
        return pd.DataFrame(stats_list)
    
    @staticmethod
    def visualize_tape(df: pd.DataFrame, isin: str, max_points: int = 10000):
        """
        Genera visualizaciones del consolidated tape.
        
        Args:
            df: Consolidated tape
            isin: ISIN para el título
            max_points: Máximo de puntos a plotear (sampling para performance)
        """
        print("\nGenerando visualizaciones...")
        
        # Sampling si el dataset es muy grande
        if len(df) > max_points:
            sample_df = df.sample(n=max_points, random_state=42).sort_values('epoch')
            print(f"  Sampling {max_points:,} puntos de {len(df):,} totales")
        else:
            sample_df = df
        
        # Extraer venues
        bid_cols = [col for col in df.columns if col.endswith('_bid')]
        venues = [col.replace('_bid', '') for col in bid_cols]
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # ====================================================================
        # Plot 1: Mid Prices por venue
        # ====================================================================
        ax1 = axes[0]
        
        for venue in venues:
            bid_col = f'{venue}_bid'
            ask_col = f'{venue}_ask'
            
            if bid_col in sample_df.columns and ask_col in sample_df.columns:
                mid_price = (sample_df[bid_col] + sample_df[ask_col]) / 2
                ax1.plot(range(len(sample_df)), mid_price, label=venue, alpha=0.7)
        
        ax1.set_title(f'Consolidated Tape - Mid Prices por Venue\nISIN: {isin}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Snapshot Index')
        ax1.set_ylabel('Mid Price (€)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # ====================================================================
        # Plot 2: Spreads por venue (basis points)
        # ====================================================================
        ax2 = axes[1]
        
        for venue in venues:
            bid_col = f'{venue}_bid'
            ask_col = f'{venue}_ask'
            
            if bid_col in sample_df.columns and ask_col in sample_df.columns:
                spread_bps = (sample_df[ask_col] - sample_df[bid_col]) * 10000
                ax2.plot(range(len(sample_df)), spread_bps, label=venue, alpha=0.7)
        
        ax2.set_title('Spreads por Venue (basis points)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Snapshot Index')
        ax2.set_ylabel('Spread (bps)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("  Visualizaciones generadas")
