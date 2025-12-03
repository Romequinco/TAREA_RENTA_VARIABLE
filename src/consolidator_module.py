"""
================================================================================
consolidator.py - Módulo de Consolidated Tape (VERSIÓN OPTIMIZADA)
================================================================================
Responsabilidades:
- Merge de datos multi-venue en un único DataFrame
- Forward fill para propagar últimos precios conocidos
- Validación del tape consolidado

OPTIMIZACIONES APLICADAS:
- Redondeo temporal a bins para alinear timestamps (evita explosión de memoria)
- merge_asof en lugar de outer merge (O(n) vs O(n²))
- Procesamiento eficiente para datasets grandes

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
    
    OPTIMIZACIÓN: Usa redondeo temporal y merge_asof para evitar explosión
    de memoria en datasets grandes.
    """
    
    def __init__(self, time_bin_ms: int = 100):
        """
        Args:
            time_bin_ms: Ventana de redondeo temporal en milisegundos (default: 100ms)
        """
        self.time_bin_ms = time_bin_ms
        self.time_bin_ns = time_bin_ms * 1_000_000  # Convertir a nanosegundos
        logger.info(f"ConsolidatedTape initialized: time_bin={time_bin_ms}ms")
    
    def _round_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Redondea timestamps a bins temporales para facilitar el merge.
        
        CRÍTICO: Esto evita la explosión combinatoria del outer merge.
        
        Args:
            df: DataFrame con columna 'epoch'
            
        Returns:
            DataFrame con 'epoch' redondeado
        """
        df = df.copy()
        df['epoch'] = (df['epoch'] // self.time_bin_ns) * self.time_bin_ns
        return df
    
    def _prepare_venue_data(self, df: pd.DataFrame, venue_name: str) -> pd.DataFrame:
        """
        Prepara datos de un venue para consolidación.
        
        Args:
            df: DataFrame con datos del venue
            venue_name: Nombre del venue (e.g., 'XMAD')
            
        Returns:
            DataFrame preparado con columnas renombradas
        """
        # Redondear timestamps
        df = self._round_timestamps(df)
        
        # Ordenar por timestamp (crítico para merge_asof)
        df = df.sort_values('epoch').reset_index(drop=True)
        
        # Renombrar columnas con prefijo del venue
        rename_map = {
            'px_bid_0': f'{venue_name}_bid',
            'px_ask_0': f'{venue_name}_ask',
            'qty_bid_0': f'{venue_name}_bid_qty',
            'qty_ask_0': f'{venue_name}_ask_qty'
        }
        
        df = df.rename(columns=rename_map)
        
        # Seleccionar solo columnas necesarias
        cols_to_keep = ['epoch'] + list(rename_map.values())
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols]
        
        # Eliminar duplicados en el mismo timestamp (tomar último)
        df = df.drop_duplicates(subset=['epoch'], keep='last')
        
        return df
    
    def create_tape(self, venue_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge de todos los venues en un único DataFrame usando estrategia optimizada.
        
        OPTIMIZACIÓN CLAVE: En lugar de outer merge (que genera producto cartesiano),
        usa merge_asof incremental con redondeo temporal.
        
        Proceso:
        1. Redondear timestamps a bins (100ms por defecto)
        2. Preparar cada venue con renombrado de columnas
        3. merge_asof incremental (empieza con venue más líquido)
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
        
        venue_names = list(venue_data.keys())
        print(f"  Venues a consolidar: {venue_names}")
        
        # ====================================================================
        # PASO 1: Preparar DataFrames individuales
        # ====================================================================
        prepared_venues = {}
        
        for mic, df in venue_data.items():
            if len(df) == 0:
                logger.warning(f"  [SKIP] {mic} no tiene datos")
                continue
            
            prepared = self._prepare_venue_data(df, mic)
            prepared_venues[mic] = prepared
            print(f"   {mic}: {len(prepared):,} snapshots preparados")
        
        if len(prepared_venues) == 0:
            logger.error("  No hay venues preparados para consolidar")
            return None
        
        # ====================================================================
        # PASO 2: Estrategia de merge eficiente
        # ====================================================================
        # Ordenar venues por cantidad de datos (más líquido primero)
        sorted_venues = sorted(
            prepared_venues.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Venue base (el más líquido)
        base_venue_name, consolidated = sorted_venues[0]
        print(f"\n  Usando {base_venue_name} como base ({len(consolidated):,} rows)")
        
        # merge_asof incremental con los demás venues
        print(f"\n  Ejecutando merge_asof incremental...")
        
        for venue_name, venue_df in sorted_venues[1:]:
            print(f"    Merging con {venue_name}... ", end='')
            
            try:
                # merge_asof: busca el timestamp más cercano
                consolidated = pd.merge_asof(
                    consolidated,
                    venue_df,
                    on='epoch',
                    direction='nearest',  # Buscar timestamp más cercano
                    tolerance=self.time_bin_ns  # Tolerancia de redondeo
                )
                print(f"OK ({len(consolidated):,} rows)")
                
            except Exception as e:
                logger.error(f"Error merging {venue_name}: {e}")
                print(f"ERROR")
                continue
        
        # ====================================================================
        # PASO 3: Ordenar por timestamp
        # ====================================================================
        consolidated = consolidated.sort_values('epoch').reset_index(drop=True)
        
        print(f"\n  [OK] Tape consolidado creado: {consolidated.shape}")
        print(f"    - Timestamps únicos: {len(consolidated):,}")
        print(f"    - Columnas totales: {len(consolidated.columns)}")
        print(f"    - Rango temporal: {consolidated['epoch'].min()} to {consolidated['epoch'].max()}")
        
        # ====================================================================
        # PASO 4: Forward fill para propagar últimos precios
        # ====================================================================
        print(f"\n  Aplicando forward fill...")
        
        nans_before = consolidated.isna().sum().sum()
        print(f"    NaNs antes: {nans_before:,}")
        
        # Forward fill: Propagar último valor conocido hacia adelante
        consolidated = consolidated.ffill()
        
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
        
        # ====================================================================
        # PASO 6: Estadísticas de cobertura
        # ====================================================================
        print(f"\n  [ESTADISTICAS DE COBERTURA]")
        for venue_name in prepared_venues.keys():
            bid_col = f'{venue_name}_bid'
            ask_col = f'{venue_name}_ask'
            
            if bid_col in consolidated.columns and ask_col in consolidated.columns:
                valid_rows = consolidated[[bid_col, ask_col]].notna().all(axis=1).sum()
                coverage = (valid_rows / len(consolidated)) * 100
                print(f"    {venue_name}: {valid_rows:,} rows válidas ({coverage:.1f}% cobertura)")
        
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
        print("  [OK] No NaNs encontrados")
        
        # ====================================================================
        # Validación 2: Timestamps monotónicos
        # ====================================================================
        if not df['epoch'].is_monotonic_increasing:
            logger.error("  [ERROR] Timestamps are not monotonically increasing")
            return False
        print("  [OK] Timestamps monotónicamente crecientes")
        
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
                    logger.error(f"  [ERROR] Negative spread found in {venue}")
                    all_spreads_valid = False
        
        if all_spreads_valid:
            print("  [OK] No negative spreads dentro de venues")
        else:
            return False
        
        # ====================================================================
        # Validación 4: No precios excesivos
        # ====================================================================
        price_cols = [col for col in df.columns if 'bid' in col or 'ask' in col]
        price_cols = [c for c in price_cols if not c.endswith('_qty')]
        
        if len(price_cols) > 0:
            max_price = df[price_cols].max().max()
            
            if max_price > config.MAX_REASONABLE_PRICE:
                logger.warning(f"  [ADVERTENCIA] Precio sospechosamente alto: €{max_price:.2f}")
            else:
                print(f"  [OK] Precios razonables (max: €{max_price:.2f})")
        
        print("\n  [EXITO] VALIDACIÓN EXITOSA")
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
        plt.savefig(config.FIGURES_DIR / f'consolidated_tape_{isin}.png', dpi=150)
        print(f"  Visualización guardada en: {config.FIGURES_DIR / f'consolidated_tape_{isin}.png'}")
        plt.close()