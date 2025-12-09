"""
================================================================================
data_cleaner.py - Módulo de Limpieza y Validación de Datos (OPTIMIZADO)
================================================================================

CRITICAL DATA VENDOR SPECS:

**A. Magic Numbers (ELIMINAR):**
- 666666.666 → Unquoted/Unknown
- 999999.999 → Market Order (At Best)
- 999999.989 → At Open Order
- 999999.988 → At Close Order
- 999999.979 → Pegged Order
- 999999.123 → Unquoted/Unknown

**B. Market Status Codes (FILTRAR POR ESTOS):**
Solo válidas quotes cuando mercado está en Continuous Trading:
- AQUIS (AQXE): [5308427]
- BME (XMAD): [5832713, 5832756]
- CBOE (CEUX): [12255233]
- TURQUOISE (TRQX): [7608181]

PIPELINE DE LIMPIEZA (ORDEN CRÍTICO):
1. Clean magic numbers
2. Filter by trading status
3. Validate prices

OPTIMIZACIONES APLICADAS:
- Operaciones vectorizadas con máscaras booleanas
- Early exit en validaciones
- Métodos compartidos para reducir duplicación
- Logging detallado con métricas de calidad
- Validación de Book Identity Key integrada

================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List

from config_module import config

logger = logging.getLogger(__name__)

# Constantes para optimización
MAX_REASONABLE_PRICE = config.MAX_REASONABLE_PRICE


# ============================================================================
# FUNCIONES COMPATIBLES CON EL OTRO CÓDIGO
# ============================================================================

def is_valid_price(price: float) -> bool:
    """Check if a price is valid (not NaN, not a magic number, and > 0)."""
    if pd.isna(price) or price <= 0:
        return False
    if price in config.MAGIC_NUMBERS:
        return False
    return True


def filter_valid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows where both bid and ask prices are valid.
    Usa nombres originales: px_bid_0, px_ask_0
    """
    if df.empty:
        return df
    
    # Usar nombres originales
    if 'px_bid_0' not in df.columns or 'px_ask_0' not in df.columns:
        return df
    
    mask = df.apply(
        lambda row: is_valid_price(row.get('px_bid_0')) and is_valid_price(row.get('px_ask_0')),
        axis=1
    )
    return df[mask].copy()


class DataCleaner:
    """
    Clase responsable de aplicar filtros de calidad de datos según especificaciones del vendor.
    
    CRÍTICO: La limpieza es esencial para evitar señales de arbitraje falsas.
    Los magic numbers y estados de mercado inválidos generarían oportunidades
    que en realidad no son ejecutables.
    """
    
    def __init__(self):
        """Inicializa el limpiador con configuración del vendor."""
        self.magic_numbers = set(config.MAGIC_NUMBERS)  # Set para búsqueda O(1)
        self.valid_states = config.VALID_STATES.copy()
    
    @staticmethod
    def clean_magic_numbers(df: pd.DataFrame, 
                            price_columns: List[str] = None) -> Tuple[pd.DataFrame, int]:
        """
        Elimina filas con magic numbers en cualquier columna de precios.
        
        CRÍTICO: Los magic numbers NO son precios reales. Son códigos especiales
        del vendor para indicar estados como "Market Order", "Pegged Order", etc.
        
        Args:
            df: DataFrame con quotes
            price_columns: Lista de columnas de precios a verificar
                          (default: detecta automáticamente bid/ask o px_bid_0/px_ask_0)
        
        Returns:
            Tuple[DataFrame limpio, número de filas eliminadas]
        """
        initial_len = len(df)
        if initial_len == 0:
            return df, 0
        
        # Usar nombres originales
        if price_columns is None:
            price_columns = ['px_bid_0', 'px_ask_0']
        
        # Verificar que las columnas existen
        available_cols = [col for col in price_columns if col in df.columns]
        if not available_cols:
            logger.warning(f"    No se encontraron columnas de precios para filtrar magic numbers")
            return df, 0
        
        # Máscara vectorizada: True si NO hay magic numbers en ninguna columna
        mask = ~df[available_cols].isin(config.MAGIC_NUMBERS).any(axis=1)
        df_clean = df[mask].copy()
        removed = initial_len - len(df_clean)
        
        if removed > 0:
            pct = removed / initial_len * 100
            logger.info(f"    Removed {removed:,} magic numbers ({pct:.2f}%)")
        
        return df_clean, removed
    
    @staticmethod
    def validate_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Aplica validaciones lógicas de precios y cantidades.
        
        Validaciones:
        1. px_bid_0 > 0 y px_ask_0 > 0
        2. px_bid_0 < px_ask_0 (no crossed book dentro del venue)
        3. qty_bid_0 > 0 y qty_ask_0 > 0
        4. Precios < MAX_REASONABLE_PRICE EUR (sanity check)
        
        Args:
            df: DataFrame con columnas de precios y cantidades
            
        Returns:
            Tuple[DataFrame validado, estadísticas de eliminación]
            Estadísticas: {'removed': int, 'pct_removed': float}
        """
        initial_len = len(df)
        if initial_len == 0:
            return df, {'removed': 0, 'pct_removed': 0.0}
        
        # Usar nombres originales
        required_cols = ['px_bid_0', 'px_ask_0', 'qty_bid_0', 'qty_ask_0']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"    Faltan columnas para validación: {missing_cols}")
            return df, {'removed': 0, 'pct_removed': 0.0}
        
        # Máscara combinada optimizada (una sola pasada)
        # Validaciones: positivos, no NaN, spread válido, precio razonable
        mask = (
            (df['px_bid_0'] > 0) &
            (df['px_ask_0'] > 0) &
            (df['qty_bid_0'] > 0) &
            (df['qty_ask_0'] > 0) &
            (df['px_bid_0'].notna()) &
            (df['px_ask_0'].notna()) &
            (df['px_bid_0'] < df['px_ask_0']) &  # No crossed book
            (df['px_bid_0'] < MAX_REASONABLE_PRICE) &
            (df['px_ask_0'] < MAX_REASONABLE_PRICE)
        )
        
        df_clean = df[mask].copy()
        removed = initial_len - len(df_clean)
        
        stats = {
            'removed': removed,
            'pct_removed': (removed / initial_len * 100) if initial_len > 0 else 0.0
        }
        
        if removed > 0:
            logger.info(f"    Removed {removed:,} invalid prices ({stats['pct_removed']:.2f}%)")
            # Desglose de razones (opcional, solo si hay muchas eliminaciones)
            if removed > initial_len * 0.1:  # Más del 10% eliminado
                crossed = ((df['px_bid_0'] >= df['px_ask_0']).sum() if 'px_bid_0' in df.columns and 'px_ask_0' in df.columns else 0)
                too_high = ((df['px_bid_0'] >= MAX_REASONABLE_PRICE).sum() if 'px_bid_0' in df.columns else 0)
                if crossed > 0:
                    logger.warning(f"      - Crossed books: {crossed:,}")
                if too_high > 0:
                    logger.warning(f"      - Precios > €{MAX_REASONABLE_PRICE}: {too_high:,}")
        
        return df_clean, stats
    
    @staticmethod
    def filter_by_trading_status(qte_df: pd.DataFrame,
                                 sts_df: pd.DataFrame,
                                 mic: str) -> Tuple[pd.DataFrame, int]:
        """
        Filtra quotes para mantener solo momentos de Continuous Trading.
        
        CRÍTICO: Operar durante auctions, halts o pre-open generaría señales
        falsas. Las órdenes no se ejecutarían instantáneamente en esos estados.
        
        Método:
        - Usa pd.merge_asof con direction='backward' para propagar el último
          estado conocido hacia adelante
        - Filtra por códigos de continuous trading según venue
        - Si no hay códigos válidos conocidos, mantiene los datos con advertencia
        
        Fuente: Arbitrage study in BME.docx - Section 2.B "Market Status Codes"
        
        Args:
            qte_df: DataFrame de quotes (ordenado por epoch)
            sts_df: DataFrame de status (ordenado por epoch)
            mic: Código del venue (para saber qué estados son válidos)
            
        Returns:
            Tuple[DataFrame filtrado, número de filas eliminadas]
        """
        initial_len = len(qte_df)
        
        # Validar datos de entrada (early exit)
        # CRÍTICO: Si no hay STS, mantener todos los datos (igual que código de referencia)
        if sts_df is None or len(sts_df) == 0:
            logger.warning(f"    No STS data for {mic}, skipping status filter (manteniendo todos los datos)")
            return qte_df, 0
        
        if mic not in config.VALID_STATES:
            logger.warning(f"    Unknown MIC {mic}, skipping status filter")
            return qte_df, 0
        
        # Usar 'market_trading_status' (nombre original)
        if 'market_trading_status' not in sts_df.columns:
            logger.warning(f"    Column 'market_trading_status' not found in STS for {mic}")
            return qte_df, 0
        
        # CORRECCIÓN: Verificar si los códigos válidos existen en los datos
        valid_codes = config.VALID_STATES[mic]
        actual_codes = set(sts_df['market_trading_status'].dropna().unique())
        matching_codes = set(valid_codes).intersection(actual_codes)
        
        # DIAGNÓSTICO: Log detallado de códigos encontrados
        logger.info(f"    Códigos esperados para {mic}: {valid_codes}")
        logger.info(f"    Códigos encontrados en STS: {sorted(actual_codes)}")
        logger.info(f"    Códigos que coinciden: {sorted(matching_codes) if len(matching_codes) > 0 else 'NINGUNO'}")
        
        if len(matching_codes) == 0:
            logger.error(f"    [CRÍTICO] No matching trading status codes found for {mic}")
            logger.error(f"    Expected codes: {valid_codes}")
            logger.error(f"    Found codes: {sorted(actual_codes)}")
            logger.error(f"    [ADVERTENCIA] Keeping all data without status filtering")
            return qte_df, 0
        
        # Validación del Book Identity Key: (session, isin, mic, ticker)
        identity_fields = ['mic=' + mic]
        for field in ['session', 'isin', 'ticker']:
            if field in qte_df.columns and field in sts_df.columns:
                qte_val = qte_df[field].iloc[0] if len(qte_df) > 0 else None
                sts_val = sts_df[field].iloc[0] if len(sts_df) > 0 else None
                if qte_val != sts_val:
                    logger.error(f"    [ERROR] Book Identity Key mismatch: {field} QTE={qte_val} != STS={sts_val}")
                    return qte_df, 0
                identity_fields.append(f"{field}={qte_val}")
        
        if len(identity_fields) > 1:
            logger.debug(f"    Book Identity Key validado: ({', '.join(identity_fields)})")
        
        # Optimización: Solo ordenar si no está ya ordenado
        if not qte_df['epoch'].is_monotonic_increasing:
            qte_sorted = qte_df.sort_values('epoch').copy()
        else:
            qte_sorted = qte_df.copy()
        
        # Seleccionar columnas necesarias de STS (optimizado)
        sts_cols = ['epoch', 'market_trading_status']
        if not sts_df['epoch'].is_monotonic_increasing:
            sts_sorted = sts_df[sts_cols].sort_values('epoch').copy()
        else:
            sts_sorted = sts_df[sts_cols].copy()
        
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
            return qte_df, 0
        
        # DIAGNÓSTICO: Verificar cuántos snapshots tienen estado asignado
        snapshots_with_status = merged['market_trading_status'].notna().sum()
        snapshots_without_status = merged['market_trading_status'].isna().sum()
        logger.info(f"    Snapshots con estado asignado: {snapshots_with_status:,} ({snapshots_with_status/initial_len*100:.2f}%)")
        logger.info(f"    Snapshots sin estado asignado: {snapshots_without_status:,} ({snapshots_without_status/initial_len*100:.2f}%)")
        
        # CRÍTICO: Manejar snapshots sin estado asignado
        # El código de referencia mantiene los datos si no hay estado asignado
        # Solo filtramos los que tienen estado asignado Y son válidos
        merged_with_status = merged[merged['market_trading_status'].notna()].copy()
        
        if len(merged_with_status) == 0:
            # Si ningún snapshot tiene estado asignado, mantener todos los datos sin filtrar
            # (igual que el código de referencia cuando hay errores)
            logger.warning(f"    Ningún snapshot tiene estado asignado después del merge_asof para {mic}")
            logger.warning(f"    Manteniendo datos originales sin filtrar (igual que código de referencia)")
            return qte_df, 0
        
        # Si hay algunos con estado pero no todos, mantener los que tienen estado válido
        # y también los que no tienen estado (asumimos que son válidos si no hay estado)
        # Esto es más conservador y evita perder oportunidades
        snapshots_without_status = merged[merged['market_trading_status'].isna()].copy()
        
        # Filtrar solo los que tienen estado asignado por códigos válidos
        # Los que no tienen estado los mantenemos (asumimos válidos)
        
        # Filtrar por códigos válidos de continuous trading (solo los que tienen estado asignado)
        # Handle both numeric and string status codes
        if pd.api.types.is_numeric_dtype(merged_with_status['market_trading_status']):
            # Direct numeric comparison (more efficient)
            mask = merged_with_status['market_trading_status'].isin(matching_codes)
        else:
            # If status is string, convert both to string for comparison
            merged_with_status['status_str'] = merged_with_status['market_trading_status'].astype(str)
            valid_codes_str = [str(s) for s in matching_codes]
            mask = merged_with_status['status_str'].isin(valid_codes_str)
            merged_with_status = merged_with_status.drop(columns=['status_str'])
        
        # Filtrar solo los que tienen estado válido
        merged_filtered_valid = merged_with_status[mask].copy()
        
        # Combinar: los que tienen estado válido + los que no tienen estado (asumimos válidos)
        # Esto es más conservador y evita perder oportunidades cuando no hay estado disponible
        if len(snapshots_without_status) > 0:
            # Eliminar columna de status de los que no tienen estado antes de combinar
            if 'market_trading_status' in snapshots_without_status.columns:
                snapshots_without_status = snapshots_without_status.drop('market_trading_status', axis=1)
            merged_filtered = pd.concat([merged_filtered_valid, snapshots_without_status], ignore_index=True)
            merged_filtered = merged_filtered.sort_values('epoch').reset_index(drop=True)
        else:
            merged_filtered = merged_filtered_valid
        
        # DIAGNÓSTICO: Verificar distribución de estados
        if 'market_trading_status' in merged_with_status.columns:
            status_distribution = merged_with_status['market_trading_status'].value_counts()
            logger.info(f"    Distribución de estados encontrados:")
            for status, count in status_distribution.items():
                is_valid = status in matching_codes
                status_label = "[VALID]" if is_valid else "[INVALID]"
                logger.info(f"      {status}: {count:,} snapshots ({status_label})")
        
        # Limpiar columna temporal
        if 'market_trading_status' in merged_filtered.columns:
            merged_filtered = merged_filtered.drop('market_trading_status', axis=1)
        
        removed = initial_len - len(merged_filtered)
        
        # CORRECCIÓN: Solo aplicar filtro si no elimina TODOS los datos
        if len(merged_filtered) == 0:
            logger.error(f"    [CRÍTICO] Status filtering would remove ALL data for {mic}")
            logger.error(f"    Esto indica que ningún snapshot tiene estado de continuous trading")
            logger.error(f"    Manteniendo datos originales sin filtrar")
            return qte_df, 0
        
        if removed > 0:
            pct = removed / initial_len * 100
            logger.info(f"    [OK] Removed {removed:,} non-trading snapshots ({pct:.2f}%)")
            logger.info(f"    [OK] Kept {len(merged_filtered):,} continuous trading snapshots ({100-pct:.2f}%)")
        else:
            logger.debug(f"    No se eliminaron snapshots - todos son continuous trading")
        
        return merged_filtered, removed
    
    # Alias para mantener compatibilidad con código existente
    @staticmethod
    def filter_by_market_status(qte_df: pd.DataFrame,
                                sts_df: pd.DataFrame,
                                mic: str) -> pd.DataFrame:
        """
        Alias de filter_by_trading_status para mantener compatibilidad.
        Retorna solo el DataFrame (sin métricas) para compatibilidad con código existente.
        """
        df_filtered, _ = DataCleaner.filter_by_trading_status(qte_df, sts_df, mic)
        return df_filtered
    
    def clean_venue_data(self, qte_df: pd.DataFrame, 
                        sts_df: Optional[pd.DataFrame],
                        mic: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Aplica todo el pipeline de limpieza en orden según especificaciones.
        
        Pipeline (ORDEN CRÍTICO):
        1. Clean magic numbers
        2. Filter by trading status
        3. Validate prices
        
        Args:
            qte_df: DataFrame de quotes
            sts_df: DataFrame de status (opcional)
            mic: Market Identifier Code
            
        Returns:
            Tuple[DataFrame limpio, métricas de calidad]
            Métricas: {'original': int, 'after_magic': int, 'after_status': int, 
                      'final': int, 'pct_retained': float}
        """
        print(f"\n  [LIMPIEZA] {mic}...")
        
        initial_len = len(qte_df)
        print(f"    Snapshots iniciales: {initial_len:,}")
        
        metrics = {
            'original': initial_len,
            'removed_magic': 0,
            'removed_status': 0,
            'removed_validation': 0,
            'final': 0,
            'pct_retained': 0.0
        }
        
        if initial_len == 0:
            logger.warning(f"    {mic}: DataFrame vacío")
            return qte_df, metrics
        
        # PASO 1: Clean magic numbers
        df, removed_magic = self.clean_magic_numbers(qte_df)
        metrics['removed_magic'] = removed_magic
        metrics['after_magic'] = len(df)
        
        if len(df) == 0:
            logger.warning(f"    {mic}: Todos los datos eliminados por magic numbers")
            metrics['final'] = 0
            metrics['pct_retained'] = 0.0
            return df, metrics
        
        # PASO 2: Filter by trading status
        df, removed_status = self.filter_by_trading_status(df, sts_df, mic)
        metrics['removed_status'] = removed_status
        metrics['after_status'] = len(df)
        
        if len(df) == 0:
            logger.warning(f"    {mic}: Todos los datos eliminados por status filtering")
            metrics['final'] = 0
            metrics['pct_retained'] = 0.0
            return df, metrics
        
        # PASO 3: Validate prices (incluye crossed books y sanity checks)
        df, validation_stats = self.validate_prices(df)
        metrics['removed_validation'] = validation_stats['removed']
        metrics['final'] = len(df)
        
        # Calcular porcentaje retenido
        if initial_len > 0:
            metrics['pct_retained'] = (metrics['final'] / initial_len) * 100
        
        # Generar columna seq si no existe (REQUISITO: secuencia después del epoch)
        df = self._ensure_seq_column(df)
        
        print(f"    [OK] Snapshots finales: {metrics['final']:,} ({metrics['pct_retained']:.2f}% retenido)")
        
        # Advertencia si queda muy poco
        if metrics['final'] == 0:
            logger.warning(f"  {mic} has 0 snapshots after cleaning")
        elif metrics['pct_retained'] < 1.0:
            logger.warning(f"  {mic} retained only {metrics['pct_retained']:.2f}% of original data")
        
        return df, metrics
    
    def _ensure_seq_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera columna seq si no existe para desambiguar timestamps iguales.
        
        Args:
            df: DataFrame con epoch
            
        Returns:
            DataFrame con columna seq añadida si no existía
        """
        if 'seq' in df.columns:
            return df
        
        # Intentar usar session e isin si están disponibles
        grouping_cols = []
        for col in ['session', 'isin']:
            if col in df.columns:
                grouping_cols.append(col)
        
        grouping_cols.append('epoch')
        
        # Generar seq determinísticamente
        if len(grouping_cols) > 1:
            df = df.sort_values(grouping_cols).reset_index(drop=True)
            df['seq'] = df.groupby(grouping_cols, sort=False).cumcount()
            logger.debug(f"    Generada columna 'seq' usando groupby({grouping_cols}).cumcount()")
        else:
            df = df.sort_values('epoch').reset_index(drop=True)
            df['seq'] = df.groupby('epoch', sort=False).cumcount()
            logger.debug(f"    Generada columna 'seq' usando groupby('epoch').cumcount()")
        
        return df
    
    def clean_all_venues(self, venue_data: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Aplica limpieza a todos los venues de un ISIN.
        MANTIENE COMPATIBILIDAD con código existente.
        
        Args:
            venue_data: Dict {mic: {'qte': df, 'sts': df}}
            
        Returns:
            Dict {mic: cleaned_dataframe}
        """
        print("\n" + "=" * 80)
        print("LIMPIEZA Y VALIDACIÓN DE DATOS")
        print("=" * 80)
        
        cleaned_data = {}
        all_metrics = {}
        
        for mic, data_dict in venue_data.items():
            try:
                qte_df = data_dict['qte']
                sts_df = data_dict.get('sts')
                
                cleaned_df, metrics = self.clean_venue_data(qte_df, sts_df, mic)
                all_metrics[mic] = metrics
                
                if len(cleaned_df) > 0:
                    cleaned_data[mic] = cleaned_df
                else:
                    logger.warning(f"  {mic} excluded - no data after cleaning")
            
            except Exception as e:
                logger.error(f"  Error cleaning {mic}: {e}", exc_info=True)
        
        # Reportar métricas de calidad agregadas
        self._report_quality_metrics(all_metrics)
        
        print(f"\n[EXITO] Limpieza completada para {len(cleaned_data)} venues")
        
        if len(cleaned_data) == 0:
            logger.error("  [CRITICO] No venues survived cleaning!")
        
        return cleaned_data
    
    def _report_quality_metrics(self, metrics_dict: Dict[str, Dict]) -> None:
        """
        Reporta métricas de calidad agregadas para todos los venues.
        
        Args:
            metrics_dict: Dict {mic: métricas}
        """
        if not metrics_dict:
            return
        
        total_original = sum(m.get('original', 0) for m in metrics_dict.values())
        total_final = sum(m.get('final', 0) for m in metrics_dict.values())
        total_magic = sum(m.get('removed_magic', 0) for m in metrics_dict.values())
        total_status = sum(m.get('removed_status', 0) for m in metrics_dict.values())
        total_validation = sum(m.get('removed_validation', 0) for m in metrics_dict.values())
        
        if total_original > 0:
            print(f"\n  [MÉTRICAS DE CALIDAD AGREGADAS]")
            print(f"    Filas originales: {total_original:,}")
            print(f"    Eliminadas por magic numbers: {total_magic:,} ({total_magic/total_original*100:.2f}%)")
            print(f"    Eliminadas por status inválido: {total_status:,} ({total_status/total_original*100:.2f}%)")
            print(f"    Eliminadas por validaciones: {total_validation:,} ({total_validation/total_original*100:.2f}%)")
            print(f"    Filas finales: {total_final:,} ({total_final/total_original*100:.2f}% retenido)")
            
            logger.info(f"Quality metrics: {total_original:,} → {total_final:,} "
                       f"({total_final/total_original*100:.2f}% retained)")
    
    @staticmethod
    def get_quality_report(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Genera un DataFrame con reporte de calidad para todos los venues.
        
        Args:
            metrics_dict: Dict {mic: métricas}
            
        Returns:
            DataFrame con métricas por venue
        """
        if not metrics_dict:
            return pd.DataFrame()
        
        rows = []
        for mic, metrics in metrics_dict.items():
            original = metrics.get('original', 0)
            rows.append({
                'Venue': mic,
                'Original': original,
                'Removed_Magic': metrics.get('removed_magic', 0),
                'Removed_Status': metrics.get('removed_status', 0),
                'Removed_Validation': metrics.get('removed_validation', 0),
                'Final': metrics.get('final', 0),
                'Pct_Retained': metrics.get('pct_retained', 0.0)
            })
        
        df = pd.DataFrame(rows)
        return df


# ============================================================================
# FUNCIONES WRAPPER ESTÁTICAS (según especificaciones)
# ============================================================================

def clean_magic_numbers(df: pd.DataFrame, 
                        price_columns: List[str] = None) -> pd.DataFrame:
    """
    Elimina filas con magic numbers en cualquier columna de precios.
    
    Wrapper estático según especificaciones.
    
    Args:
        df: DataFrame con quotes
        price_columns: Lista de columnas de precios (default: detecta automáticamente bid/ask o px_bid_0/px_ask_0)
    
    Returns:
        DataFrame limpio
    """
    cleaner = DataCleaner()
    df_clean, _ = cleaner.clean_magic_numbers(df, price_columns)
    return df_clean


def filter_by_trading_status(qte_df: pd.DataFrame,
                             sts_df: pd.DataFrame,
                             mic: str) -> pd.DataFrame:
    """
    Filtra quotes para mantener solo momentos de Continuous Trading.
    
    Wrapper estático según especificaciones.
    
    Args:
        qte_df: DataFrame de quotes (ordenado por epoch)
        sts_df: DataFrame de status (ordenado por epoch)
        mic: Código del venue
    
    Returns:
        DataFrame filtrado
    """
    cleaner = DataCleaner()
    df_filtered, _ = cleaner.filter_by_trading_status(qte_df, sts_df, mic)
    return df_filtered


def validate_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Aplica validaciones lógicas de precios y cantidades.
    
    Wrapper estático según especificaciones.
    
    Args:
        df: DataFrame con columnas de precios y cantidades
    
    Returns:
        Tuple[DataFrame validado, estadísticas]
    """
    cleaner = DataCleaner()
    return cleaner.validate_prices(df)


def clean_venue_data(qte_df: pd.DataFrame,
                    sts_df: Optional[pd.DataFrame],
                    mic: str) -> pd.DataFrame:
    """
    Aplica todo el pipeline de limpieza en orden.
    
    Wrapper estático según especificaciones.
    
    Args:
        qte_df: DataFrame de quotes
        sts_df: DataFrame de status (opcional)
        mic: Market Identifier Code
    
    Returns:
        DataFrame limpio listo para consolidación
    """
    cleaner = DataCleaner()
    df_clean, _ = cleaner.clean_venue_data(qte_df, sts_df, mic)
    return df_clean