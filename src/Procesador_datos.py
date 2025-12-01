"""
================================================================================
PROYECTO: High-Frequency Arbitrage Detection in Fragmented Markets
PARTES 1 + 2: Config + Loader + Cleaner + CONSOLIDATED TAPE
================================================================================

Objetivo: Sistema completo de carga, limpieza y consolidación de datos
          multi-venue para detección de arbitraje
================================================================================
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de visualizaciones
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("INICIANDO SISTEMA DE DETECCIÓN DE ARBITRAJE HFT")
print("=" * 80)

# ============================================================================
# MÓDULO 1: CONFIGURACIÓN GLOBAL
# ============================================================================

class Config:
    """Configuración centralizada del proyecto"""
    
    # Directorios
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_SMALL_DIR = DATA_DIR / "DATA_SMALL"
    DATA_BIG_DIR = DATA_DIR / "DATA_BIG"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    
    # Magic Numbers - CRÍTICO para calidad de datos
    MAGIC_NUMBERS = [
        666666.666, 999999.999, 999999.989, 
        999999.988, 999999.979, 999999.123
    ]
    
    # Market Status Codes - Solo continuous trading
    VALID_STATES = {
        'XMAD': [5832713, 5832756],  # BME
        'AQXE': [5308427],            # AQUIS
        'CEUX': [12255233],           # CBOE
        'TRQX': [7608181]             # TURQUOISE
    }
    
    # Latency buckets (microsegundos)
    LATENCY_BUCKETS = [
        0, 100, 500, 1000, 2000, 3000, 4000, 5000,
        10000, 15000, 20000, 30000, 50000, 100000
    ]
    
    # Thresholds
    MIN_PROFIT_PER_UNIT = 0.0001
    MIN_THEORETICAL_PROFIT = 0.10
    MAX_REASONABLE_PRICE = 10000
    SUSPICIOUS_PROFIT_THRESHOLD = 1000
    
    # Performance
    CHUNK_SIZE = 100000
    N_JOBS = -1

config = Config()
print("\n✓ Configuración cargada correctamente")

# ============================================================================
# MÓDULO 2: DATA LOADER
# ============================================================================

print("\n" + "=" * 80)
print("MÓDULO 2: DATA LOADER")
print("=" * 80)

class DataLoader:
    """Carga de archivos QTE (Quotes) y STS (Status)"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        logger.info(f"DataLoader initialized: {data_dir}")
    
    def discover_isins(self) -> List[str]:
        """Descubre todos los ISINs únicos en el directorio"""
        print("\n🔍 Descubriendo ISINs disponibles...")
        
        qte_files = glob.glob(str(self.data_dir / "**" / "QTE_*.csv.gz"), recursive=True)
        
        isins = set()
        for file in qte_files:
            parts = Path(file).stem.split('_')
            if len(parts) >= 5:
                isin = parts[2]
                isins.add(isin)
        
        isins_list = sorted(list(isins))
        print(f"  ✓ Encontrados {len(isins_list)} ISINs únicos")
        
        return isins_list
    
    def load_qte_file(self, filepath: Path) -> pd.DataFrame:
        """Carga un archivo QTE (order book snapshots)"""
        try:
            df = pd.read_csv(filepath, compression='gzip')
            
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
        """Carga un archivo STS (trading status changes)"""
        try:
            df = pd.read_csv(filepath, compression='gzip')
            
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
        Carga todos los datos (QTE + STS) para un ISIN.
        
        Returns:
            {
                'XMAD': {'qte': DataFrame, 'sts': DataFrame},
                'AQXE': {'qte': DataFrame, 'sts': DataFrame},
                ...
            }
        """
        print(f"\n Cargando datos para ISIN: {isin}")
        
        qte_pattern = f"QTE_*_{isin}_*.csv.gz"
        sts_pattern = f"STS_*_{isin}_*.csv.gz"
        
        qte_files = glob.glob(str(self.data_dir / "**" / qte_pattern), recursive=True)
        sts_files = glob.glob(str(self.data_dir / "**" / sts_pattern), recursive=True)
        
        print(f"  - QTE files: {len(qte_files)}")
        print(f"  - STS files: {len(sts_files)}")
        
        venue_data = {}
        
        for qte_file in qte_files:
            parts = Path(qte_file).stem.split('_')
            mic = parts[4] if len(parts) >= 5 else None
            
            if mic is None:
                continue
            
            qte_df = self.load_qte_file(Path(qte_file))
            if qte_df is None:
                continue
            
            sts_file = qte_file.replace('QTE_', 'STS_')
            sts_df = self.load_sts_file(Path(sts_file)) if Path(sts_file).exists() else None
            
            venue_data[mic] = {
                'qte': qte_df,
                'sts': sts_df
            }
            
            print(f"    ✓ {mic}: {len(qte_df):,} snapshots")
        
        print(f"  ✓ Loaded {len(venue_data)} venues")
        return venue_data

loader = DataLoader(config.DATA_SMALL_DIR)

# ============================================================================
# MÓDULO 3: DATA CLEANER
# ============================================================================

print("\n" + "=" * 80)
print("MÓDULO 3: DATA CLEANER")
print("=" * 80)

class DataCleaner:
    """Pipeline de limpieza de datos"""
    
    @staticmethod
    def clean_magic_numbers(df: pd.DataFrame) -> pd.DataFrame:
        """Elimina snapshots con magic numbers"""
        initial_len = len(df)
        
        bid_mask = ~df['px_bid_0'].isin(config.MAGIC_NUMBERS)
        ask_mask = ~df['px_ask_0'].isin(config.MAGIC_NUMBERS)
        
        df_clean = df[bid_mask & ask_mask].copy()
        
        removed = initial_len - len(df_clean)
        if removed > 0:
            pct = removed / initial_len * 100
            logger.info(f"  Removed {removed:,} magic numbers ({pct:.1f}%)")
        
        return df_clean
    
    @staticmethod
    def clean_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Valida precios y cantidades positivos"""
        initial_len = len(df)
        
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
            logger.info(f"  Removed {removed:,} invalid prices")
        
        return df_clean
    
    @staticmethod
    def clean_crossed_book(df: pd.DataFrame) -> pd.DataFrame:
        """Elimina crossed books (bid >= ask en mismo venue)"""
        initial_len = len(df)
        
        mask = df['px_bid_0'] < df['px_ask_0']
        df_clean = df[mask].copy()
        
        removed = initial_len - len(df_clean)
        if removed > 0:
            logger.warning(f"  Removed {removed:,} crossed books")
        
        return df_clean
    
    @staticmethod
    def filter_by_market_status(qte_df: pd.DataFrame, 
                                sts_df: pd.DataFrame, 
                                mic: str) -> pd.DataFrame:
        """
        Filtra para mantener solo continuous trading.
        Usa merge_asof para propagar último estado conocido.
        """
        if sts_df is None or len(sts_df) == 0:
            logger.warning(f"  No STS data for {mic}")
            return qte_df
        
        if mic not in config.VALID_STATES:
            logger.warning(f"  Unknown MIC {mic}")
            return qte_df
        
        initial_len = len(qte_df)
        
        qte_sorted = qte_df.sort_values('epoch')
        sts_sorted = sts_df[['epoch', 'market_trading_status']].sort_values('epoch')
        
        merged = pd.merge_asof(
            qte_sorted,
            sts_sorted,
            on='epoch',
            direction='backward'
        )
        
        valid_codes = config.VALID_STATES[mic]
        merged_filtered = merged[merged['market_trading_status'].isin(valid_codes)].copy()
        
        if 'market_trading_status' in merged_filtered.columns:
            merged_filtered = merged_filtered.drop('market_trading_status', axis=1)
        
        removed = initial_len - len(merged_filtered)
        if removed > 0:
            pct = removed / initial_len * 100
            logger.info(f"  Removed {removed:,} non-trading ({pct:.1f}%)")
        
        return merged_filtered
    
    def clean_venue_data(self, venue_dict: Dict, mic: str) -> pd.DataFrame:
        """Pipeline completo para un venue"""
        print(f"\n  Limpiando {mic}...")
        
        qte_df = venue_dict['qte']
        sts_df = venue_dict['sts']
        
        print(f"    - Inicial: {len(qte_df):,} snapshots")
        
        df = qte_df.copy()
        df = self.clean_magic_numbers(df)
        df = self.clean_invalid_prices(df)
        df = self.clean_crossed_book(df)
        df = self.filter_by_market_status(df, sts_df, mic)
        
        print(f"    ✓ Final: {len(df):,} snapshots limpios")
        
        return df
    
    def clean_all_venues(self, venue_data: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """Limpia todos los venues de un ISIN"""
        cleaned_data = {}
        
        for mic, data_dict in venue_data.items():
            try:
                cleaned_df = self.clean_venue_data(data_dict, mic)
                
                if len(cleaned_df) > 0:
                    cleaned_data[mic] = cleaned_df
                else:
                    logger.warning(f"  {mic} has 0 snapshots after cleaning")
            
            except Exception as e:
                logger.error(f"  Error cleaning {mic}: {e}")
        
        return cleaned_data

cleaner = DataCleaner()

# ============================================================================
# MÓDULO 4: CONSOLIDATED TAPE (NUEVO - PARTE 2)
# ============================================================================

print("\n" + "=" * 80)
print("MÓDULO 4: CONSOLIDATED TAPE")
print("=" * 80)

class ConsolidatedTape:
    """
    Crea un DataFrame unificado con precios de todos los venues.
    
    Objetivo: Comparar precios cross-venue en el mismo instante temporal
              para detectar oportunidades de arbitraje.
    
    Estructura del tape:
        epoch | XMAD_bid | XMAD_ask | XMAD_bid_qty | XMAD_ask_qty |
              | AQXE_bid | AQXE_ask | AQXE_bid_qty | AQXE_ask_qty |
              | ...
    """
    
    @staticmethod
    def create_tape(venue_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge de todos los venues en un único DataFrame.
        
        Método:
        1. Outer join por timestamp (epoch)
        2. Forward fill para propagar últimos precios
        3. Validaciones de calidad
        """
        print("\n🔗 Creando Consolidated Tape...")
        
        if len(venue_data) == 0:
            logger.error(" No venue data to consolidate")
            return None
        
        print(f"  - Venues a consolidar: {list(venue_data.keys())}")
        
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
            print(f"    ✓ {mic}: {len(df_subset):,} snapshots preparados")
        
        # Merge iterativo con outer join
        print("\n  Ejecutando outer merge...")
        consolidated = dfs[0]
        
        for i, df in enumerate(dfs[1:], 1):
            consolidated = pd.merge(
                consolidated,
                df,
                on='epoch',
                how='outer'
            )
            print(f"    - Merge {i}/{len(dfs)-1} completado")
        
        # Ordenar por timestamp
        consolidated = consolidated.sort_values('epoch').reset_index(drop=True)
        
        print(f"\n  ✓ Tape consolidado: {consolidated.shape}")
        print(f"    - Timestamps únicos: {len(consolidated):,}")
        print(f"    - Columnas: {len(consolidated.columns)}")
        print(f"    - Rango temporal: {consolidated['epoch'].min()} to {consolidated['epoch'].max()}")
        
        # Forward fill: Propagar últimos precios conocidos
        print("\n  Aplicando forward fill...")
        nans_before = consolidated.isna().sum().sum()
        print(f"    - NaNs antes: {nans_before:,}")
        
        consolidated = consolidated.fillna(method='ffill')
        
        nans_after = consolidated.isna().sum().sum()
        print(f"    - NaNs después: {nans_after:,}")
        
        # Eliminar primeras filas con NaNs (antes del primer update de cada venue)
        initial_nans = consolidated.isna().sum(axis=1)
        first_complete_row = (initial_nans == 0).idxmax()
        
        if first_complete_row > 0:
            print(f"    - Eliminando primeras {first_complete_row} filas incompletas")
            consolidated = consolidated.iloc[first_complete_row:].reset_index(drop=True)
        
        print(f"\n  Tape final: {consolidated.shape}")
        
        return consolidated
    
    @staticmethod
    def validate_tape(df: pd.DataFrame) -> bool:
        """
        Validaciones críticas del consolidated tape.
        
        Validaciones:
        1. No NaNs
        2. Timestamps monotónicos
        3. No negative spreads por venue
        4. No precios excesivos
        """
        print("\nValidando Consolidated Tape...")
        
        # 1. No NaNs
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.error(f" Found {nan_count:,} NaNs")
            return False
        print("  No NaNs encontrados")
        
        # 2. Timestamps monotónicos
        if not df['epoch'].is_monotonic_increasing:
            logger.error(" Timestamps not monotonic")
            return False
        print(" Timestamps monotónicamente crecientes")
        
        # 3. No negative spreads
        bid_cols = [col for col in df.columns if col.endswith('_bid')]
        
        all_spreads_valid = True
        for bid_col in bid_cols:
            ask_col = bid_col.replace('_bid', '_ask')
            
            if ask_col in df.columns:
                spread = df[ask_col] - df[bid_col]
                
                if (spread < 0).any():
                    venue = bid_col.replace('_bid', '')
                    logger.error(f" Negative spread in {venue}")
                    all_spreads_valid = False
        
        if all_spreads_valid:
            print(" No negative spreads dentro de venues")
        else:
            return False
        
        # 4. No precios excesivos
        price_cols = [col for col in df.columns if 'bid' in col or 'ask' in col]
        max_price = df[price_cols].max().max()
        
        if max_price > config.MAX_REASONABLE_PRICE:
            logger.warning(f"  Precio sospechosamente alto: €{max_price:.2f}")
        else:
            print(f"  Precios razonables (max: €{max_price:.2f})")
        
        print("\n  VALIDACIÓN EXITOSA")
        return True
    
    @staticmethod
    def visualize_tape(df: pd.DataFrame, isin: str, max_points: int = 10000):
        """
        Visualiza el consolidated tape para análisis.
        """
        print("\n Generando visualizaciones...")
        
        # Limitar puntos para performance
        if len(df) > max_points:
            sample_df = df.sample(n=max_points).sort_values('epoch')
        else:
            sample_df = df
        
        # Extraer venues
        bid_cols = [col for col in df.columns if col.endswith('_bid')]
        venues = [col.replace('_bid', '') for col in bid_cols]
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Precios por venue
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
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spreads por venue
        ax2 = axes[1]
        for venue in venues:
            bid_col = f'{venue}_bid'
            ask_col = f'{venue}_ask'
            
            if bid_col in sample_df.columns and ask_col in sample_df.columns:
                spread = (sample_df[ask_col] - sample_df[bid_col]) * 10000  # En basis points
                ax2.plot(range(len(sample_df)), spread, label=venue, alpha=0.7)
        
        ax2.set_title('Spreads por Venue (basis points)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Snapshot Index')
        ax2.set_ylabel('Spread (bps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("  Visualizaciones generadas")

tape_builder = ConsolidatedTape()

# ============================================================================
# EJECUCIÓN COMPLETA - PARTES 1 + 2
# ============================================================================

print("\n" + "=" * 80)
print("EJECUTANDO PIPELINE COMPLETO (PARTES 1 + 2)")
print("=" * 80)

# Paso 1: Descubrir ISINs
isins = loader.discover_isins()

if len(isins) == 0:
    print("\nERROR: No ISINs encontrados")
else:
    # Paso 2: Seleccionar ISIN de prueba
    test_isin = isins[0]
    print(f"\nISIN seleccionado: {test_isin}")
    
    # Paso 3: Cargar datos raw
    print("\n" + "=" * 80)
    print("FASE 1: CARGA DE DATOS RAW")
    print("=" * 80)
    raw_data = loader.load_isin_data(test_isin)
    
    # Paso 4: Limpiar datos
    print("\n" + "=" * 80)
    print("FASE 2: LIMPIEZA Y VALIDACIÓN")
    print("=" * 80)
    clean_data = cleaner.clean_all_venues(raw_data)
    
    # Paso 5: Crear consolidated tape
    print("\n" + "=" * 80)
    print("FASE 3: CONSOLIDATED TAPE")
    print("=" * 80)
    consolidated_tape = tape_builder.create_tape(clean_data)
    
    # Paso 6: Validar tape
    if consolidated_tape is not None:
        is_valid = tape_builder.validate_tape(consolidated_tape)
        
        if is_valid:
            # Paso 7: Visualizar
            tape_builder.visualize_tape(consolidated_tape, test_isin)
            
            # Paso 8: Estadísticas finales
            print("\n" + "=" * 80)
            print("ESTADÍSTICAS DEL CONSOLIDATED TAPE")
            print("=" * 80)
            
            venues = [col.replace('_bid', '') for col in consolidated_tape.columns 
                     if col.endswith('_bid')]
            
            for venue in venues:
                bid_col = f'{venue}_bid'
                ask_col = f'{venue}_ask'
                
                if bid_col in consolidated_tape.columns:
                    print(f"\n{venue}:")
                    print(f"  - Bid medio: €{consolidated_tape[bid_col].mean():.4f}")
                    print(f"  - Ask medio: €{consolidated_tape[ask_col].mean():.4f}")
                    
                    spread = consolidated_tape[ask_col] - consolidated_tape[bid_col]
                    print(f"  - Spread medio: €{spread.mean():.4f}")
                    print(f"  - Spread min/max: €{spread.min():.4f} / €{spread.max():.4f}")

print("\n" + "=" * 80)
print("PARTES 1 + 2 COMPLETADAS CON ÉXITO")
print("=" * 80)
print("\nPróximos pasos:")
print("  - PARTE 3: Signal Generation (detección Bid_max > Ask_min)")
print("  - PARTE 4: Latency Simulation (time machine)")
print("  - PARTE 5: Análisis Final + Visualizaciones")

print("\nVariables disponibles para la siguiente parte:")
print("  - config: Configuración global")
print("  - loader: DataLoader")
print("  - cleaner: DataCleaner")
print("  - tape_builder: ConsolidatedTape")
print("  - isins: Lista de ISINs")
print("  - clean_data: Dict[mic, DataFrame] con datos limpios")
print("  - consolidated_tape: DataFrame con el tape consolidado ⭐")
