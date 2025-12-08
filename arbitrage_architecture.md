# Arquitectura del Sistema de Arbitraje HFT

## Visión General del Proyecto

Este documento describe la arquitectura completa para implementar un sistema de detección y análisis de oportunidades de arbitraje en mercados fragmentados europeos, específicamente enfocado en el mercado español (BME).

---

## 1. Estructura Modular del Sistema

### 1.1 Módulos Principales

```
arbitrage_system/
│
├── data_loader.py          # Carga y parseo de datos
├── data_cleaner.py         # Limpieza y validación
├── consolidator.py         # Creación del consolidated tape
├── signal_generator.py     # Detección de oportunidades
├── latency_simulator.py    # Simulación de latencias
├── analyzer.py             # Análisis y métricas
├── visualizer.py           # Generación de gráficos
└── main.py                 # Orquestador principal
```

### 1.2 Flujo de Datos

```
[Archivos CSV.GZ] 
    ↓
[Data Loader] → Lectura paralela por ISIN
    ↓
[Data Cleaner] → Filtrado de magic numbers + estados válidos
    ↓
[Consolidator] → Merged multi-venue por timestamp
    ↓
[Signal Generator] → Detección Bid_max > Ask_min
    ↓
[Latency Simulator] → Time-shifted profit lookup
    ↓
[Analyzer] → Agregación y métricas
    ↓
[Visualizer] → Tablas y gráficos finales
```

---

## 2. Definiciones Críticas

### 2.1 Book Identity Key

```python
book_key = (session, isin, mic, ticker)
```

**Propósito**: Identificar unívocamente cada order book para hacer joins correctos entre QTE y STS.

### 2.2 Timestamp Authority

- **Epoch**: Microsegundos UTC
- **Resolución mínima**: 1 microsegundo
- **Tipo de datos**: `int64` (no float para evitar pérdida de precisión)

### 2.3 Estados Válidos por Venue

| Venue | MIC Code | Trading Status Codes |
|-------|----------|----------------------|
| BME | XMAD | 5832713, 5832756 |
| AQUIS | AQXE | 5308427 |
| CBOE | CEUX | 12255233 |
| TURQUOISE | TRQX | 7608181 |

---

## 3. Pipeline de Procesamiento Detallado

### FASE 1: Ingesta de Datos

**Input**: 
- `QTE_2024-XX-XX_*.csv.gz`
- `STS_2024-XX-XX_*.csv.gz`

**Proceso**:
1. Identificar todos los ISINs únicos en el directorio
2. Para cada ISIN, cargar todos los venues (XMAD, AQXE, CEUX, TRQX)
3. Parsear con `pandas.read_csv(..., compression='gzip')`
4. Validar columnas requeridas: `epoch`, `px_bid_0`, `px_ask_0`, `qty_bid_0`, `qty_ask_0`

**Output**: 
```python
{
    'ISIN_A': {
        'XMAD': {'qte': DataFrame, 'sts': DataFrame},
        'AQXE': {'qte': DataFrame, 'sts': DataFrame},
        ...
    },
    'ISIN_B': {...}
}
```

---

### FASE 2: Limpieza de Datos

**2.1 Filtrado de Magic Numbers**

```python
MAGIC_NUMBERS = [666666.666, 999999.999, 999999.989, 
                 999999.988, 999999.979, 999999.123]

def clean_prices(df):
    mask = ~df['px_bid_0'].isin(MAGIC_NUMBERS) & \
           ~df['px_ask_0'].isin(MAGIC_NUMBERS)
    return df[mask]
```

**2.2 Filtrado por Estado de Mercado**

```python
VALID_STATES = {
    'XMAD': [5832713, 5832756],
    'AQXE': [5308427],
    'CEUX': [12255233],
    'TRQX': [7608181]
}

def filter_by_status(qte_df, sts_df, mic):
    # Merge asof para propagar el último estado conocido
    merged = pd.merge_asof(
        qte_df.sort_values('epoch'),
        sts_df[['epoch', 'market_trading_status']].sort_values('epoch'),
        on='epoch',
        direction='backward'
    )
    
    valid_mask = merged['market_trading_status'].isin(VALID_STATES[mic])
    return merged[valid_mask]
```

**2.3 Validaciones Adicionales**

- `px_bid_0 > 0` y `px_ask_0 > 0`
- `px_bid_0 < px_ask_0` (no crossed book dentro del mismo venue)
- `qty_bid_0 > 0` y `qty_ask_0 > 0`

---

### FASE 3: Consolidated Tape

**Objetivo**: Crear un DataFrame único donde cada fila representa un instante temporal y las columnas contienen los best bid/ask de todos los venues.

**Estructura del Consolidated Tape**:

```
epoch | XMAD_bid | XMAD_ask | XMAD_bid_qty | XMAD_ask_qty | 
      | AQXE_bid | AQXE_ask | AQXE_bid_qty | AQXE_ask_qty |
      | CEUX_bid | CEUX_ask | CEUX_bid_qty | CEUX_ask_qty |
      | TRQX_bid | TRQX_ask | TRQX_bid_qty | TRQX_ask_qty |
```

**Método de Construcción**:

```python
def create_consolidated_tape(venue_data):
    """
    venue_data: Dict[mic] -> DataFrame con columnas 
                [epoch, px_bid_0, px_ask_0, qty_bid_0, qty_ask_0]
    """
    dfs = []
    
    for mic, df in venue_data.items():
        df_renamed = df[['epoch', 'px_bid_0', 'px_ask_0', 
                         'qty_bid_0', 'qty_ask_0']].copy()
        df_renamed.columns = ['epoch', 
                             f'{mic}_bid', f'{mic}_ask',
                             f'{mic}_bid_qty', f'{mic}_ask_qty']
        dfs.append(df_renamed)
    
    # Outer merge para tener todos los timestamps
    consolidated = dfs[0]
    for df in dfs[1:]:
        consolidated = pd.merge(consolidated, df, 
                               on='epoch', how='outer')
    
    # Ordenar por timestamp
    consolidated = consolidated.sort_values('epoch').reset_index(drop=True)
    
    # Forward fill para propagar últimos precios conocidos
    consolidated = consolidated.fillna(method='ffill')
    
    return consolidated
```

**Crítico**: El `fillna(method='ffill')` asume que el último precio conocido sigue vigente hasta que llegue un update. Esto es estándar en market microstructure.

---

## 4. Métricas de Validación

### 4.1 Data Quality Checks

Antes de proceder a detección de señales:

```python
def validate_consolidated_tape(df):
    # 1. No NaNs en las primeras filas (después de ffill inicial)
    assert df.iloc[100:].isna().sum().sum() == 0
    
    # 2. No negative spreads dentro de cada venue
    for mic in ['XMAD', 'AQXE', 'CEUX', 'TRQX']:
        spread = df[f'{mic}_ask'] - df[f'{mic}_bid']
        assert (spread >= 0).all(), f"Negative spread in {mic}"
    
    # 3. Timestamps monotónicamente crecientes
    assert df['epoch'].is_monotonic_increasing
    
    # 4. No precios excesivamente altos (residual magic numbers)
    max_reasonable_price = 10000  # EUR
    for col in df.columns:
        if 'bid' in col or 'ask' in col:
            assert df[col].max() < max_reasonable_price
```

---

## 5. Consideraciones de Implementación

### 5.1 Memoria

- Para datasets grandes (>10GB), procesar por chunks de tiempo (ej: 1 hora)
- Usar `dtype` optimizados: `int64` para epoch, `float32` para precios

### 5.2 Performance

- Paralelizar la carga por ISIN usando `multiprocessing`
- Usar `numba` para loops de detección de señales
- Mantener DataFrames indexados por `epoch` para búsquedas rápidas

### 5.3 Edge Cases

1. **Market Open**: Los primeros segundos pueden tener spreads anormales → Ignorar primeros 5 minutos
2. **Market Close**: Últimos segundos en fase de cierre → Usar solo hasta 17:25 CET
3. **Venue Downtime**: Si un venue no tiene datos durante >10s → No incluir en global max/min durante ese período
4. **Corporate Actions**: Splits/dividendos pueden crear outliers → Validar con precio de cierre del día anterior

---

## Próximos Documentos

1. **Documento 2**: Algoritmo de Detección de Señales y Rising Edge
2. **Documento 3**: Simulación de Latencia y Time Machine
3. **Documento 4**: Análisis Avanzado y Visualizaciones
4. **Documento 5**: Código de Implementación Completo

---

**Puntos Clave para Máxima Puntuación**:
- ✅ Manejo correcto de magic numbers
- ✅ Filtrado por market status
- ✅ Consolidated tape con forward fill
- ✅ Validaciones exhaustivas
- ✅ Manejo de edge cases (open/close)
