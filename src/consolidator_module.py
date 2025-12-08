"""
================================================================================
consolidator.py - Módulo de Consolidated Tape (OPTIMIZADO)
================================================================================

OBJETIVO:
Crear un DataFrame donde cada fila = un timestamp único, y las columnas contienen
el best bid/ask de TODOS los venues simultáneamente.

ESTRUCTURA DEL CONSOLIDATED TAPE:
| epoch      | XMAD_bid | XMAD_ask | XMAD_bid_qty | XMAD_ask_qty |
|            | AQXE_bid | AQXE_ask | AQXE_bid_qty | AQXE_ask_qty |
|            | CEUX_bid | CEUX_ask | CEUX_bid_qty | CEUX_ask_qty |
|            | TRQX_bid | TRQX_ask | TRQX_bid_qty | TRQX_ask_qty |

ALGORITMO:
1. Renombrar columnas por venue
2. Merge iterativo (outer merge o merge_asof optimizado)
3. Ordenar por timestamp
4. Forward Fill (CRÍTICO)
5. Validaciones exhaustivas

OPTIMIZACIONES APLICADAS:
- merge_asof para mejor rendimiento (O(n) vs O(n²))
- Redondeo temporal opcional para datasets grandes
- Validaciones exhaustivas según especificaciones
- Manejo robusto de edge cases

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

from config_module import config

logger = logging.getLogger(__name__)


class ConsolidatedTape:
    """
    Crea y valida el Consolidated Tape multi-venue.
    
    El tape consolidado permite comparar precios de todos los venues en
    el mismo instante temporal, que es esencial para detectar arbitraje.
    """
    
    def __init__(self, time_bin_ms: Optional[int] = None):
        """
        Args:
            time_bin_ms: Ventana de redondeo temporal en milisegundos (None = sin redondeo)
        """
        self.time_bin_ms = time_bin_ms
        if time_bin_ms:
            self.time_bin_ns = time_bin_ms * 1_000_000  # Convertir a nanosegundos
        else:
            self.time_bin_ns = None
        logger.info(f"ConsolidatedTape initialized: time_bin={time_bin_ms}ms" if time_bin_ms else "ConsolidatedTape initialized: sin redondeo temporal")
    
    @staticmethod
    def _rename_venue_columns(df: pd.DataFrame, venue_name: str) -> pd.DataFrame:
        """
        Renombra columnas por venue según especificaciones.
        
        Renombramientos:
        - px_bid_0 → {MIC}_bid
        - px_ask_0 → {MIC}_ask
        - qty_bid_0 → {MIC}_bid_qty
        - qty_ask_0 → {MIC}_ask_qty
        
        Args:
            df: DataFrame con datos del venue
            venue_name: Nombre del venue (e.g., 'XMAD')
            
        Returns:
            DataFrame con columnas renombradas
        """
        rename_map = {
            'px_bid_0': f'{venue_name}_bid',
            'px_ask_0': f'{venue_name}_ask',
            'qty_bid_0': f'{venue_name}_bid_qty',
            'qty_ask_0': f'{venue_name}_ask_qty'
        }
        
        # Solo renombrar columnas que existen
        existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        
        if not existing_rename:
            logger.warning(f"  No se encontraron columnas estándar en {venue_name}")
            return df
        
        df = df.rename(columns=existing_rename)
        
        # Seleccionar solo columnas necesarias
        cols_to_keep = ['epoch'] + list(existing_rename.values())
        available_cols = [c for c in cols_to_keep if c in df.columns]
        
        return df[available_cols]
    
    def _round_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Redondea timestamps a bins temporales (opcional, para datasets grandes).
        
        Args:
            df: DataFrame con columna 'epoch'
            
        Returns:
            DataFrame con 'epoch' redondeado (si time_bin está configurado)
        """
        if self.time_bin_ns is None:
            return df
        
        df = df.copy()
        df['epoch'] = (df['epoch'] // self.time_bin_ns) * self.time_bin_ns
        return df
    
    def _prepare_venue_data(self, df: pd.DataFrame, venue_name: str) -> pd.DataFrame:
        """
        Prepara datos de un venue para consolidación.
        
        Args:
            df: DataFrame con datos del venue
            venue_name: Nombre del venue
            
        Returns:
            DataFrame preparado
        """
        # Renombrar columnas
        df = self._rename_venue_columns(df, venue_name)
        
        # Redondear timestamps si está configurado
        df = self._round_timestamps(df)
        
        # Ordenar por timestamp (crítico para merge)
        df = df.sort_values('epoch').reset_index(drop=True)
        
        # Eliminar duplicados en el mismo timestamp (tomar último)
        df = df.drop_duplicates(subset=['epoch'], keep='last')
        
        return df
    
    def create_tape(self, venue_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Crea el consolidated tape usando merge iterativo optimizado.
        
        Proceso:
        1. Renombrar columnas por venue
        2. Merge iterativo (merge_asof optimizado o outer merge)
        3. Ordenar por timestamp
        4. Forward Fill (CRÍTICO)
        5. Eliminar filas iniciales con NaNs
        
        Args:
            venue_data: Dict[mic] -> DataFrame con columnas:
                       ['epoch', 'px_bid_0', 'px_ask_0', 'qty_bid_0', 'qty_ask_0']
            
        Returns:
            DataFrame consolidado con columnas:
            ['epoch', 'XMAD_bid', 'XMAD_ask', 'XMAD_bid_qty', 'XMAD_ask_qty', ...]
        """
        print("\n" + "=" * 80)
        print("CREANDO CONSOLIDATED TAPE")
        print("=" * 80)
        
        if len(venue_data) == 0:
            logger.error("  No venue data to consolidate")
            return None
        
        venue_names = list(venue_data.keys())
        print(f"  Venues a consolidar: {venue_names}")
        
        # PASO 1: Preparar DataFrames individuales
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
        
        # PASO 2: Merge iterativo
        # Estrategia: usar merge_asof si hay redondeo temporal (más eficiente),
        # sino usar outer merge según especificaciones
        
        if self.time_bin_ns is not None:
            # Estrategia optimizada: merge_asof incremental
            sorted_venues = sorted(
                prepared_venues.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            base_venue_name, consolidated = sorted_venues[0]
            print(f"\n  Usando {base_venue_name} como base ({len(consolidated):,} rows)")
            print(f"  Ejecutando merge_asof incremental...")
            
            for venue_name, venue_df in sorted_venues[1:]:
                print(f"    Merging con {venue_name}... ", end='')
                try:
                    consolidated = pd.merge_asof(
                        consolidated.sort_values('epoch'),
                        venue_df.sort_values('epoch'),
                        on='epoch',
                        direction='backward',  # REQUISITO: backward para propagar último valor conocido
                        tolerance=self.time_bin_ns
                    )
                    print(f"OK ({len(consolidated):,} rows)")
                except Exception as e:
                    logger.error(f"Error merging {venue_name}: {e}")
                    print(f"ERROR")
                    continue
        else:
            # Estrategia según especificaciones: outer merge iterativo
            print(f"\n  Ejecutando outer merge iterativo...")
            
            sorted_venues = sorted(
                prepared_venues.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            base_venue_name, consolidated = sorted_venues[0]
            print(f"  Usando {base_venue_name} como base ({len(consolidated):,} rows)")
            
            for venue_name, venue_df in sorted_venues[1:]:
                print(f"    Merging con {venue_name}... ", end='')
                try:
                    consolidated = pd.merge(
                        consolidated,
                        venue_df,
                        on='epoch',
                        how='outer',
                        suffixes=('', '_dup')
                    )
                    # Eliminar columnas duplicadas si las hay
                    consolidated = consolidated.loc[:, ~consolidated.columns.str.endswith('_dup')]
                    print(f"OK ({len(consolidated):,} rows)")
                except Exception as e:
                    logger.error(f"Error merging {venue_name}: {e}")
                    print(f"ERROR")
                    continue
        
        # PASO 3: Ordenar por timestamp
        consolidated = consolidated.sort_values('epoch').reset_index(drop=True)
        
        print(f"\n  [OK] Tape consolidado creado: {consolidated.shape}")
        print(f"    - Timestamps únicos: {len(consolidated):,}")
        print(f"    - Columnas totales: {len(consolidated.columns)}")
        
        # PASO 4: Forward Fill (CRÍTICO)
        # CRÍTICO: Asunción de market microstructure - el último precio conocido sigue vigente
        # hasta que llegue un nuevo update. Esto es estándar en análisis de order books.
        # Sin forward fill, tendríamos NaNs en cada timestamp donde un venue no actualiza,
        # lo cual haría imposible comparar precios entre venues.
        print(f"\n  Aplicando forward fill...")
        nans_before = consolidated.isna().sum().sum()
        print(f"    NaNs antes: {nans_before:,}")
        
        # Forward fill: Propagar último valor conocido hacia adelante
        # Ejemplo: Si XMAD actualiza en T=100 y T=200, el precio en T=150 será el de T=100
        consolidated = consolidated.ffill()
        
        nans_after = consolidated.isna().sum().sum()
        print(f"    NaNs después: {nans_after:,}")
        
        # PASO 5: Eliminar primeras filas con NaNs
        # Opción implementada: Eliminar filas iniciales incompletas
        initial_nans = consolidated.isna().sum(axis=1)
        
        if initial_nans.max() > 0:
            # Encontrar primera fila completa
            first_complete_row = (initial_nans == 0).idxmax() if (initial_nans == 0).any() else 0
            
            if first_complete_row > 0:
                print(f"    Eliminando primeras {first_complete_row} filas incompletas")
                consolidated = consolidated.iloc[first_complete_row:].reset_index(drop=True)
        
        print(f"\n  Tape final: {consolidated.shape}")
        
        # Estadísticas de cobertura
        print(f"\n  [ESTADISTICAS DE COBERTURA]")
        for venue_name in prepared_venues.keys():
            bid_col = f'{venue_name}_bid'
            ask_col = f'{venue_name}_ask'
            
            if bid_col in consolidated.columns and ask_col in consolidated.columns:
                valid_rows = consolidated[[bid_col, ask_col]].notna().all(axis=1).sum()
                coverage = (valid_rows / len(consolidated)) * 100 if len(consolidated) > 0 else 0
                print(f"    {venue_name}: {valid_rows:,} rows válidas ({coverage:.1f}% cobertura)")
        
        return consolidated
    
    @staticmethod
    def validate_consolidated_tape(df: pd.DataFrame) -> bool:
        """
        Valida el consolidated tape según especificaciones.
        
        Verifica:
        1. No hay NaNs después de las primeras 100 filas (post ffill)
        2. No hay spreads negativos dentro de cada venue: {MIC}_ask - {MIC}_bid >= 0
        3. Timestamps son monotónicos crecientes
        4. No hay precios > MAX_REASONABLE_PRICE EUR (residual magic numbers)
        
        Args:
            df: Consolidated tape
            
        Returns:
            True si todas las validaciones pasan
            
        Raises:
            AssertionError con mensaje descriptivo si falla alguna validación
        """
        print("\n" + "=" * 80)
        print("VALIDANDO CONSOLIDATED TAPE")
        print("=" * 80)
        
        if df is None or len(df) == 0:
            raise AssertionError("Consolidated tape está vacío")
        
        # Validación 1: No NaNs después de las primeras 100 filas
        if len(df) > 100:
            post_initial = df.iloc[100:]
            nan_count = post_initial.isna().sum().sum()
            if nan_count > 0:
                raise AssertionError(f"Encontrados {nan_count:,} NaNs después de las primeras 100 filas (post ffill)")
        print("  [OK] No NaNs después de las primeras 100 filas")
        
        # Validación 2: Timestamps monotónicos
        if not df['epoch'].is_monotonic_increasing:
            raise AssertionError("Timestamps no son monotónicamente crecientes")
        print("  [OK] Timestamps monotónicamente crecientes")
        
        # Validación 3: No spreads negativos dentro de cada venue
        bid_cols = [col for col in df.columns if col.endswith('_bid') and not col.endswith('_bid_qty')]
        
        for bid_col in bid_cols:
            ask_col = bid_col.replace('_bid', '_ask')
            
            if ask_col in df.columns:
                spread = df[ask_col] - df[bid_col]
                negative_spreads = (spread < 0).sum()
                
                if negative_spreads > 0:
                    venue = bid_col.replace('_bid', '')
                    raise AssertionError(f"Encontrados {negative_spreads:,} spreads negativos en {venue}")
        
        print("  [OK] No spreads negativos dentro de venues")
        
        # Validación 4: No precios excesivos (residual magic numbers)
        price_cols = [col for col in df.columns if ('bid' in col or 'ask' in col) and not col.endswith('_qty')]
        
        if price_cols:
            max_price = df[price_cols].max().max()
            
            if max_price > config.MAX_REASONABLE_PRICE:
                raise AssertionError(f"Precio sospechosamente alto encontrado: €{max_price:.2f} (posible magic number residual)")
            
            print(f"  [OK] Precios razonables (max: €{max_price:.2f})")
        
        print("\n  [EXITO] VALIDACIÓN EXITOSA")
        return True
    
    # Alias para mantener compatibilidad con código existente
    @staticmethod
    def validate_tape(df: pd.DataFrame) -> bool:
        """
        Alias de validate_consolidated_tape para mantener compatibilidad.
        """
        return ConsolidatedTape.validate_consolidated_tape(df)
    
    @staticmethod
    def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """Calcula estadísticas descriptivas del consolidated tape."""
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS DEL CONSOLIDATED TAPE")
        print("=" * 80)
        
        bid_cols = [col for col in df.columns if col.endswith('_bid') and not col.endswith('_bid_qty')]
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
        
        return pd.DataFrame(stats_list)
    
    @staticmethod
    def visualize_tape(df: pd.DataFrame, isin: str, max_points: int = 10000):
        """Genera visualizaciones del consolidated tape."""
        print("\nGenerando visualizaciones...")
        
        if len(df) > max_points:
            sample_df = df.sample(n=max_points, random_state=42).sort_values('epoch')
            print(f"  Sampling {max_points:,} puntos de {len(df):,} totales")
        else:
            sample_df = df.copy()
        
        bid_cols = [col for col in df.columns if col.endswith('_bid') and not col.endswith('_bid_qty')]
        venues = [col.replace('_bid', '') for col in bid_cols]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        x_values = range(len(sample_df))
        
        # Plot 1: Mid Prices por venue
        ax1 = axes[0]
        
        for venue in venues:
            bid_col = f'{venue}_bid'
            ask_col = f'{venue}_ask'
            
            if bid_col in sample_df.columns and ask_col in sample_df.columns:
                mid_price = (sample_df[bid_col] + sample_df[ask_col]) / 2
                valid_mask = mid_price.notna()
                
                if valid_mask.any():
                    ax1.plot(
                        [x for x, v in zip(x_values, valid_mask) if v],
                        mid_price[valid_mask].values, 
                        label=venue, 
                        alpha=0.7,
                        linewidth=1.5
                    )
        
        ax1.set_title(f'Consolidated Tape - Mid Prices por Venue\nISIN: {isin}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Snapshot Index')
        ax1.set_ylabel('Mid Price (€)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spreads por venue
        ax2 = axes[1]
        
        for venue in venues:
            bid_col = f'{venue}_bid'
            ask_col = f'{venue}_ask'
            
            if bid_col in sample_df.columns and ask_col in sample_df.columns:
                spread = sample_df[ask_col] - sample_df[bid_col]
                spread_bps = spread * 10000
                valid_mask = spread_bps.notna() & (spread_bps >= 0)
                
                if valid_mask.any():
                    ax2.plot(
                        [x for x, v in zip(x_values, valid_mask) if v],
                        spread_bps[valid_mask].values, 
                        label=venue, 
                        alpha=0.7,
                        linewidth=1.5
                    )
        
        ax2.set_title('Spreads por Venue (basis points)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Snapshot Index')
        ax2.set_ylabel('Spread (bps)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = config.FIGURES_DIR / f'consolidated_tape_{isin}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Visualización guardada en: {output_path}")
        
        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception as e:
            logger.warning(f"  No se pudo mostrar gráfica interactiva: {e}")


# ============================================================================
# FUNCIÓN WRAPPER ESTÁTICA (según especificaciones)
# ============================================================================

def create_consolidated_tape(venue_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Función principal para crear el consolidated tape.
    
    Wrapper estático según especificaciones.
    
    Args:
        venue_data: Dict[mic] -> DataFrame con columnas:
                   ['epoch', 'px_bid_0', 'px_ask_0', 'qty_bid_0', 'qty_ask_0']
    
    Returns:
        DataFrame consolidado con columnas:
        ['epoch', 'XMAD_bid', 'XMAD_ask', 'XMAD_bid_qty', 'XMAD_ask_qty', ...]
    """
    tape_builder = ConsolidatedTape(time_bin_ms=None)  # Sin redondeo por defecto
    return tape_builder.create_tape(venue_data)
