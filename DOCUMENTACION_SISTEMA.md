# DOCUMENTACIÓN COMPLETA DEL SISTEMA DE ARBITRAJE HFT

## Índice

1. [Introducción](#introducción)
2. [Arquitectura General](#arquitectura-general)
3. [Flujo de Datos Completo](#flujo-de-datos-completo)
4. [Módulos del Sistema](#módulos-del-sistema)
   - [Data Loader](#1-data-loader-module)
   - [Data Cleaner](#2-data-cleaner-module)
   - [Consolidator](#3-consolidator-module)
   - [Signal Generator](#4-signal-generator-module)
   - [Latency Simulator](#5-latency-simulator-module)
   - [Analyzer](#6-analyzer-module)
5. [Estructuras de Datos](#estructuras-de-datos)
6. [Algoritmos Clave](#algoritmos-clave)
7. [Ejemplos Prácticos](#ejemplos-prácticos)

---

## Introducción

Este sistema detecta oportunidades de arbitraje estadístico en mercados fragmentados europeos (BME, AQUIS, CBOE, TURQUOISE) analizando datos de order books en tiempo real.

**Objetivo Principal:** Identificar instantes donde `Global Max Bid > Global Min Ask`, es decir, donde se puede comprar en un venue y vender en otro simultáneamente con profit positivo.

**Preguntas que Responde:**
1. ¿Existen oportunidades de arbitraje?
2. ¿Cuál es el profit teórico máximo?
3. ¿Cómo decae el profit con la latencia?

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE ARBITRAJE HFT                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  1. DATA LOADER                     │
        │  Carga archivos QTE/STS            │
        │  Formato: CSV.GZ comprimido         │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  2. DATA CLEANER                    │
        │  Filtra magic numbers               │
        │  Filtra por trading status          │
        │  Valida precios y cantidades        │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  3. CONSOLIDATOR                    │
        │  Crea consolidated tape             │
        │  Merge multi-venue                  │
        │  Forward fill de precios            │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  4. SIGNAL GENERATOR                │
        │  Detecta arbitraje                  │
        │  Calcula profits teóricos           │
        │  Rising edge detection              │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  5. LATENCY SIMULATOR               │
        │  Simula impacto de latencia         │
        │  "Time Machine" approach            │
        │  Calcula profit real                │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  6. ANALYZER                        │
        │  Genera métricas agregadas           │
        │  Crea reportes                      │
        │  Visualizaciones                    │
        └─────────────────────────────────────┘
```

---

## Flujo de Datos Completo

### Paso 1: Carga de Datos (Data Loader)

**Input:** Archivos CSV.GZ con formato:
```
QTE_2024-01-15_ES0113900J37_SAN_XMAD_1.csv.gz
STS_2024-01-15_ES0113900J37_SAN_XMAD_1.csv.gz
```

**Proceso Interno:**

1. **Parsing de Filenames:**
   ```python
   # Formato: <type>_<session>_<isin>_<ticker>_<mic>_<part>.csv.gz
   # Ejemplo: QTE_2024-01-15_ES0113900J37_SAN_XMAD_1.csv.gz
   parts = filename.split('_')
   # parts[0] = 'QTE'
   # parts[1] = '2024-01-15' (session)
   # parts[2] = 'ES0113900J37' (isin)
   # parts[3] = 'SAN' (ticker)
   # parts[4] = 'XMAD' (mic)
   # parts[5] = '1' (part)
   ```

2. **Lectura de Archivos:**
   ```python
   # Intenta múltiples encodings: utf-8, latin-1, cp1252
   df = pd.read_csv(
       filepath,
       compression='gzip',
       sep=';',              # Separador: punto y coma
       decimal='.',          # Decimal: punto (formato anglosajón)
       dtype={'epoch': 'int64'},  # CRÍTICO: int64 para precisión
       low_memory=False
   )
   ```

3. **Estructura de Datos QTE (Quotes):**
   ```python
   # Columnas requeridas:
   {
       'epoch': int64,           # Timestamp en nanosegundos UTC
       'px_bid_0': float64,      # Precio bid nivel 0 (mejor bid)
       'px_ask_0': float64,     # Precio ask nivel 0 (mejor ask)
       'qty_bid_0': float64,    # Cantidad bid nivel 0
       'qty_ask_0': float64,    # Cantidad ask nivel 0
       # Opcionales: niveles 1-9 (px_bid_1, px_ask_1, etc.)
   }
   ```

4. **Estructura de Datos STS (Status):**
   ```python
   # Columnas requeridas:
   {
       'epoch': int64,                    # Timestamp en nanosegundos UTC
       'market_trading_status': int64     # Código de estado del mercado
   }
   ```

5. **Book Identity Key:**
   ```python
   # Identificador único de cada order book:
   book_key = (session, isin, mic, ticker)
   # Ejemplo: ('2024-01-15', 'ES0113900J37', 'XMAD', 'SAN')
   ```

**Output:** Diccionario estructurado:
```python
{
    'XMAD': {
        'qte': DataFrame,  # Order book snapshots
        'sts': DataFrame   # Trading status changes
    },
    'AQXE': {
        'qte': DataFrame,
        'sts': DataFrame
    },
    ...
}
```

---

### Paso 2: Limpieza de Datos (Data Cleaner)

**Pipeline de Limpieza (ORDEN CRÍTICO):**

#### 2.1. Clean Magic Numbers

**Problema:** Los archivos contienen valores especiales que NO son precios reales:
- `666666.666` → Unquoted/Unknown
- `999999.999` → Market Order (At Best)
- `999999.989` → At Open Order
- `999999.988` → At Close Order
- `999999.979` → Pegged Order
- `999999.123` → Unquoted/Unknown

**Proceso Interno:**
```python
# Crear máscara: True si NO hay magic numbers
mask = pd.Series(True, index=df.index)
for col in ['px_bid_0', 'px_ask_0']:
    mask &= ~df[col].isin(MAGIC_NUMBERS)  # O(1) lookup con set

df_clean = df[mask].copy()
```

**Ejemplo:**
```
Antes:
  epoch          px_bid_0      px_ask_0
  1000000000     10.50         10.52
  1000001000     999999.999    10.52    ← Magic number (eliminar)
  1000002000     10.51         10.53

Después:
  epoch          px_bid_0      px_ask_0
  1000000000     10.50         10.52
  1000002000     10.51         10.53
```

#### 2.2. Filter by Trading Status

**Problema:** Solo queremos datos durante "Continuous Trading". Otros estados (auctions, halts, pre-open) generan señales falsas.

**Códigos Válidos por Venue:**
```python
VALID_STATES = {
    'XMAD': [5832713, 5832756],  # BME
    'AQXE': [5308427],            # AQUIS
    'CEUX': [12255233],           # CBOE
    'TRQX': [7608181]             # TURQUOISE
}
```

**Proceso Interno (merge_asof backward):**
```python
# 1. Ordenar ambos DataFrames por epoch
qte_sorted = qte_df.sort_values('epoch')
sts_sorted = sts_df.sort_values('epoch')

# 2. Merge asof con direction='backward'
# Esto asigna a cada snapshot el ÚLTIMO estado conocido
merged = pd.merge_asof(
    qte_sorted,
    sts_sorted[['epoch', 'market_trading_status']],
    on='epoch',
    direction='backward'  # CRÍTICO: usar último estado conocido
)

# 3. Filtrar solo continuous trading
valid_mask = merged['market_trading_status'].isin(VALID_STATES[mic])
df_filtered = merged[valid_mask]
```

**Ejemplo Visual:**
```
STS (Status Changes):
  epoch          market_trading_status
  1000000000     5832713  ← Continuous Trading
  1000005000     5832700  ← Auction (NO válido)
  1000010000     5832713  ← Continuous Trading

QTE (Quotes):
  epoch          px_bid_0    Estado Asignado
  1000002000     10.50       5832713  ← Válido (último estado conocido)
  1000007000     10.51       5832700  ← Inválido (auction)
  1000012000     10.52       5832713  ← Válido

Resultado Final:
  epoch          px_bid_0
  1000002000     10.50
  1000012000     10.52
```

#### 2.3. Validate Prices

**Validaciones Aplicadas:**
```python
mask = (
    (df['px_bid_0'] > 0) &                    # Precios positivos
    (df['px_ask_0'] > 0) &
    (df['qty_bid_0'] > 0) &                   # Cantidades positivas
    (df['qty_ask_0'] > 0) &
    (df['px_bid_0'] < df['px_ask_0']) &       # No crossed book
    (df['px_bid_0'] < MAX_REASONABLE_PRICE) & # Sanity check (< €10,000)
    (df['px_ask_0'] < MAX_REASONABLE_PRICE)
)
```

**Ejemplo de Crossed Book (eliminado):**
```
  epoch          px_bid_0    px_ask_0    Estado
  1000000000     10.52       10.50       ← Crossed (eliminar)
  1000001000     10.50       10.52       ← Válido
```

#### 2.4. Generar Columna `seq`

**Problema:** Múltiples eventos pueden tener el mismo `epoch`. Necesitamos desambiguar.

**Solución:**
```python
# Si no existe columna 'seq', generarla determinísticamente
if 'seq' not in df.columns:
    df['seq'] = df.groupby(['session', 'isin', 'epoch']).cumcount()
```

**Ejemplo:**
```
Antes:
  epoch          px_bid_0
  1000000000     10.50
  1000000000     10.51    ← Mismo epoch
  1000000000     10.52    ← Mismo epoch

Después:
  epoch          seq      px_bid_0
  1000000000     0        10.50
  1000000000     1        10.51
  1000000000     2        10.52
```

**Output:** DataFrame limpio listo para consolidación

---

### Paso 3: Consolidación (Consolidator)

**Objetivo:** Crear un único DataFrame donde cada fila = timestamp único, y las columnas contienen precios de TODOS los venues simultáneamente.

#### 3.1. Renombrar Columnas por Venue

**Proceso:**
```python
# Para cada venue, renombrar columnas:
rename_map = {
    'px_bid_0': 'XMAD_bid',
    'px_ask_0': 'XMAD_ask',
    'qty_bid_0': 'XMAD_bid_qty',
    'qty_ask_0': 'XMAD_ask_qty'
}
```

**Ejemplo:**
```
Venue XMAD:
  epoch          px_bid_0    px_ask_0
  1000000000     10.50       10.52

Después de renombrar:
  epoch          XMAD_bid    XMAD_ask
  1000000000     10.50       10.52
```

#### 3.2. Merge Iterativo

**Estrategia:** Outer merge para incluir todos los timestamps únicos.

**Proceso:**
```python
# Empezar con el venue con más datos
base_venue = sorted_venues[0]  # Ej: XMAD
consolidated = base_venue_df

# Merge con cada venue adicional
for venue_df in sorted_venues[1:]:
    consolidated = pd.merge(
        consolidated,
        venue_df,
        on='epoch',
        how='outer'  # Incluir todos los timestamps
    )
```

**Ejemplo Visual:**
```
XMAD:
  epoch          XMAD_bid    XMAD_ask
  1000000000     10.50       10.52
  1000002000     10.51       10.53

AQXE:
  epoch          AQXE_bid    AQXE_ask
  1000001000     10.49       10.51
  1000003000     10.52       10.54

Después de Outer Merge:
  epoch          XMAD_bid    XMAD_ask    AQXE_bid    AQXE_ask
  1000000000     10.50       10.52       NaN         NaN
  1000001000     NaN         NaN         10.49       10.51
  1000002000     10.51       10.53       NaN         NaN
  1000003000     NaN         NaN         10.52       10.54
```

#### 3.3. Forward Fill (CRÍTICO)

**Problema:** Tenemos NaNs donde un venue no actualiza en un timestamp específico.

**Solución:** Forward fill - asumir que el último precio conocido sigue vigente.

**Proceso:**
```python
# Ordenar por timestamp
consolidated = consolidated.sort_values('epoch')

# Forward fill: propagar último valor conocido hacia adelante
consolidated = consolidated.ffill()
```

**Ejemplo Visual:**
```
Antes de Forward Fill:
  epoch          XMAD_bid    XMAD_ask    AQXE_bid    AQXE_ask
  1000000000     10.50       10.52       NaN         NaN
  1000001000     NaN         NaN         10.49       10.51
  1000002000     10.51       10.53       NaN         NaN

Después de Forward Fill:
  epoch          XMAD_bid    XMAD_ask    AQXE_bid    AQXE_ask
  1000000000     10.50       10.52       NaN         NaN
  1000001000     10.50       10.52       10.49       10.51  ← XMAD propagado
  1000002000     10.51       10.53       10.49       10.51  ← AQXE propagado
```

**Justificación:** En market microstructure, el último precio conocido sigue vigente hasta que llegue un nuevo update. Esto es estándar en análisis de order books.

#### 3.4. Eliminar Filas Iniciales Incompletas

**Proceso:**
```python
# Encontrar primera fila donde todos los venues tienen datos
first_complete_row = (consolidated.isna().sum(axis=1) == 0).idxmax()
consolidated = consolidated.iloc[first_complete_row:]
```

**Output:** Consolidated Tape listo para detección de señales

**Estructura Final:**
```python
{
    'epoch': int64,
    'XMAD_bid': float64,
    'XMAD_ask': float64,
    'XMAD_bid_qty': float64,
    'XMAD_ask_qty': float64,
    'AQXE_bid': float64,
    'AQXE_ask': float64,
    ...
}
```

---

### Paso 4: Detección de Señales (Signal Generator)

**Objetivo:** Detectar instantes donde `Global Max Bid > Global Min Ask`

#### 4.1. Calcular Global Max Bid y Global Min Ask

**Proceso:**
```python
# Extraer columnas de bids y asks
bid_cols = [col for col in df.columns if col.endswith('_bid') and not col.endswith('_bid_qty')]
ask_cols = [col for col in df.columns if col.endswith('_ask') and not col.endswith('_ask_qty')]

# Calcular máximos y mínimos de forma vectorizada
df['max_bid'] = df[bid_cols].max(axis=1)  # Mejor bid entre todos los venues
df['min_ask'] = df[ask_cols].min(axis=1)  # Mejor ask entre todos los venues

# Identificar venues
df['venue_max_bid'] = df[bid_cols].idxmax(axis=1).str.replace('_bid', '')
df['venue_min_ask'] = df[ask_cols].idxmin(axis=1).str.replace('_ask', '')
```

**Ejemplo:**
```
Consolidated Tape:
  epoch          XMAD_bid    XMAD_ask    AQXE_bid    AQXE_ask
  1000000000     10.50       10.52       10.49       10.51

Cálculo:
  max_bid = max(10.50, 10.49) = 10.50  ← XMAD
  min_ask = min(10.52, 10.51) = 10.51  ← AQXE
  
  Condición: 10.50 > 10.51? NO → No hay arbitraje
```

**Ejemplo con Arbitraje:**
```
Consolidated Tape:
  epoch          XMAD_bid    XMAD_ask    AQXE_bid    AQXE_ask
  1000000000     10.52       10.54       10.49       10.51

Cálculo:
  max_bid = max(10.52, 10.49) = 10.52  ← XMAD
  min_ask = min(10.54, 10.51) = 10.51  ← AQXE
  
  Condición: 10.52 > 10.51? SÍ → ¡ARBITRAJE!
  
  Profit por unidad: 10.52 - 10.51 = €0.01
```

#### 4.2. Extraer Cantidades

**Proceso:**
```python
# Extraer cantidades de los venues específicos
for venue in venues:
    bid_qty_col = f'{venue}_bid_qty'
    ask_qty_col = f'{venue}_ask_qty'
    
    mask_bid = df['venue_max_bid'] == venue
    mask_ask = df['venue_min_ask'] == venue
    
    df.loc[mask_bid, 'bid_qty'] = df.loc[mask_bid, bid_qty_col]
    df.loc[mask_ask, 'ask_qty'] = df.loc[mask_ask, ask_qty_col]

# Cantidad ejecutable = mínimo entre bid y ask
df['executable_qty'] = np.minimum(df['bid_qty'], df['ask_qty'])
```

**Ejemplo:**
```
Oportunidad detectada:
  venue_max_bid = 'XMAD'
  venue_min_ask = 'AQXE'
  XMAD_bid_qty = 500
  AQXE_ask_qty = 300
  
  executable_qty = min(500, 300) = 300
```

#### 4.3. Calcular Profit Teórico

**Proceso:**
```python
# Profit por unidad
df['theoretical_profit'] = df['max_bid'] - df['min_ask']

# Profit total = Profit por unidad × Cantidad ejecutable
df['total_profit'] = df['theoretical_profit'] * df['executable_qty']
```

**Ejemplo:**
```
theoretical_profit = 10.52 - 10.51 = €0.01 por unidad
executable_qty = 300 unidades
total_profit = 0.01 × 300 = €3.00
```

#### 4.4. Rising Edge Detection (CRÍTICO)

**Problema:** Si una oportunidad persiste durante 1000 snapshots, no queremos contarla 1000 veces.

**Solución:** Solo contar la PRIMERA aparición de cada oportunidad continua.

**Proceso:**
```python
# Comparar con snapshot anterior
prev_signal = df['signal'].shift(1, fill_value=0)

# Rising edge = señal actual (1) AND no había señal anterior (0)
df['is_rising_edge'] = (df['signal'] == 1) & (prev_signal == 0)
```

**Ejemplo Visual:**
```
Snapshot    signal    prev_signal    is_rising_edge
1           0         0             False
2           1         0             True   ← Primera aparición
3           1         1             False  ← Continúa
4           1         1             False  ← Continúa
5           0         1             False  ← Desaparece
6           1         0             True   ← Nueva oportunidad
```

**Output:** DataFrame con señales detectadas y profits calculados

---

### Paso 5: Simulación de Latencia (Latency Simulator)

**Concepto Clave - "Time Machine":** Si detecto una oportunidad en T, pero mi sistema tiene latencia Δ, ejecuto en T+Δ. Para entonces, el mercado puede haber cambiado.

#### 5.1. Preparar Consolidated Tape para Búsqueda

**Proceso:**
```python
# Indexar por epoch para búsqueda binaria O(log n)
consolidated_sorted = consolidated_df.sort_values('epoch')
consolidated_indexed = consolidated_sorted.set_index('epoch')
```

#### 5.2. Para Cada Señal: Buscar Estado Futuro

**Proceso:**
```python
for signal in signals:
    signal_epoch = signal['epoch']
    execution_epoch = signal_epoch + latency_ns  # T + Δ
    
    # Búsqueda binaria eficiente O(log n)
    idx = consolidated_indexed.index.searchsorted(execution_epoch, side='left')
    
    # Obtener estado futuro del mercado
    future_row = consolidated_indexed.iloc[idx]
    
    # Recalcular max_bid y min_ask en el futuro
    actual_max_bid = future_row[bid_cols].max()
    actual_min_ask = future_row[ask_cols].min()
    
    # Profit actualizado
    actual_profit = max(0, actual_max_bid - actual_min_ask)
```

**Ejemplo Visual:**
```
Señal detectada en T=1000000000:
  max_bid = 10.52
  min_ask = 10.51
  theoretical_profit = €0.01

Latencia: 1000 us (1 ms)
Execution epoch: 1000001000

Estado del mercado en T+Δ:
  XMAD_bid = 10.51  ← Bajó
  AQXE_ask = 10.52  ← Subió
  
  actual_max_bid = 10.51
  actual_min_ask = 10.52
  actual_profit = max(0, 10.51 - 10.52) = €0.00  ← Oportunidad desapareció
```

#### 5.3. Calcular Profit Decay

**Proceso:**
```python
profit_decay = ((theoretical_profit - actual_profit) / theoretical_profit) * 100
```

**Ejemplo:**
```
theoretical_profit = €0.01
actual_profit = €0.00
profit_decay = ((0.01 - 0.00) / 0.01) * 100 = 100%  ← Pérdida total
```

**Output:** DataFrame con profits reales considerando latencia

---

### Paso 6: Análisis y Reportes (Analyzer)

#### 6.1. Resumen por ISIN

**Proceso:**
```python
for isin in signals_dict:
    rising_edges = signals_dict[isin][signals_dict[isin]['is_rising_edge']]
    
    summary = {
        'isin': isin,
        'total_opportunities': len(rising_edges),
        'theoretical_profit_0lat': rising_edges['total_profit'].sum(),
        'actual_profit_100us': latency_profits.get(100, 0.0),
        'actual_profit_1ms': latency_profits.get(1000, 0.0),
        ...
    }
```

#### 6.2. Money Table

**Formato:**
```
             0µs     100µs   1ms    10ms   100ms
ISIN_A       1234€   1100€   890€   320€   45€
ISIN_B       567€    520€    410€   120€   12€
...
TOTAL        5678€   5123€   3456€  980€   123€
```

#### 6.3. Curva de Decay

**Proceso:**
```python
# Agregar todos los ISINs por latencia
decay_curve = combined_df.groupby('latency_us').agg({
    'total_theoretical_profit': 'sum',
    'total_actual_profit': 'sum',
    ...
})

# Calcular porcentaje del teórico
decay_curve['profit_pct_of_theoretical'] = (
    decay_curve['total_actual_profit'] / 
    decay_curve['total_theoretical_profit'] * 100
)
```

**Output:** Métricas agregadas y reportes completos

---

## Estructuras de Datos

### Book Identity Key

**Definición:**
```python
book_key = (session, isin, mic, ticker)
```

**Propósito:** Identificar unívocamente cada order book para joins QTE-STS correctos.

**Ejemplo:**
```python
book_key = ('2024-01-15', 'ES0113900J37', 'XMAD', 'SAN')
```

### Epoch (Timestamp)

**Formato:** `int64` (nanosegundos UTC)

**Conversión:**
```python
# Nanosegundos a datetime
datetime = pd.to_datetime(epoch, unit='ns')

# Ejemplo:
epoch = 1705320000000000000  # Nanosegundos
datetime = 2024-01-15 09:00:00 UTC
```

**CRÍTICO:** Debe ser `int64` para evitar pérdida de precisión y errores en operaciones temporales.

### Magic Numbers

**Valores Especiales (NO son precios reales):**
```python
MAGIC_NUMBERS = [
    666666.666,  # Unquoted/Unknown
    999999.999,  # Market Order (At Best)
    999999.989,  # At Open Order
    999999.988,  # At Close Order
    999999.979,  # Pegged Order
    999999.123   # Unquoted/Unknown
]
```

**Uso:** Filtrar estos valores antes de cualquier análisis.

---

## Algoritmos Clave

### 1. Merge AsOf Backward

**Propósito:** Asignar el último estado conocido a cada snapshot.

**Algoritmo:**
```python
merged = pd.merge_asof(
    qte_sorted,      # DataFrame de quotes (ordenado por epoch)
    sts_sorted,      # DataFrame de status (ordenado por epoch)
    on='epoch',
    direction='backward'  # Usar último valor conocido
)
```

**Complejidad:** O(n log n) donde n = número de filas

### 2. Rising Edge Detection

**Propósito:** Identificar primera aparición de cada oportunidad continua.

**Algoritmo:**
```python
prev_signal = df['signal'].shift(1, fill_value=0)
is_rising_edge = (df['signal'] == 1) & (prev_signal == 0)
```

**Complejidad:** O(n) donde n = número de snapshots

### 3. Búsqueda Binaria (Time Machine)

**Propósito:** Encontrar estado futuro del mercado eficientemente.

**Algoritmo:**
```python
idx = consolidated_indexed.index.searchsorted(execution_epoch, side='left')
future_row = consolidated_indexed.iloc[idx]
```

**Complejidad:** O(log n) donde n = número de timestamps únicos

### 4. Forward Fill

**Propósito:** Propagar último precio conocido hacia adelante.

**Algoritmo:**
```python
consolidated = consolidated.ffill()
```

**Complejidad:** O(n) donde n = número de filas

---

## Ejemplos Prácticos

### Ejemplo 1: Detección de Arbitraje Simple

**Input:**
```
Consolidated Tape (timestamp único):
  XMAD_bid = 10.52, XMAD_ask = 10.54
  AQXE_bid = 10.49, AQXE_ask = 10.51
```

**Proceso:**
1. `max_bid = max(10.52, 10.49) = 10.52` (XMAD)
2. `min_ask = min(10.54, 10.51) = 10.51` (AQXE)
3. `10.52 > 10.51` → ¡ARBITRAJE DETECTADO!
4. `theoretical_profit = 10.52 - 10.51 = €0.01`
5. `executable_qty = min(XMAD_bid_qty, AQXE_ask_qty)`
6. `total_profit = theoretical_profit × executable_qty`

**Output:** Oportunidad registrada con profit calculado

### Ejemplo 2: Impacto de Latencia

**Señal Original (T=0):**
```
max_bid = 10.52
min_ask = 10.51
theoretical_profit = €0.01
```

**Latencia:** 1000 us (1 ms)

**Estado Futuro (T+1000us):**
```
max_bid = 10.51  ← Bajó
min_ask = 10.52  ← Subió
actual_profit = max(0, 10.51 - 10.52) = €0.00
```

**Resultado:** Oportunidad desapareció debido a latencia

### Ejemplo 3: Rising Edge Detection

**Snapshots:**
```
1: signal=0
2: signal=1  ← Rising edge (primera aparición)
3: signal=1  ← Continúa (no contar)
4: signal=1  ← Continúa (no contar)
5: signal=0  ← Desaparece
6: signal=1  ← Rising edge (nueva oportunidad)
```

**Resultado:** Solo snapshots 2 y 6 se cuentan como oportunidades únicas

---

## Consideraciones de Rendimiento

### Optimizaciones Aplicadas

1. **Búsqueda Binaria:** O(log n) en lugar de O(n) para búsquedas temporales
2. **Operaciones Vectorizadas:** Uso de pandas/numpy en lugar de loops Python
3. **Sets para Lookups:** O(1) para verificación de magic numbers
4. **Reducción de Copias:** Solo copiar DataFrames cuando es necesario modificar
5. **Early Exit:** Validaciones tempranas para evitar procesamiento innecesario

### Complejidad Temporal

- **Carga de Datos:** O(n) donde n = número de filas
- **Limpieza:** O(n) donde n = número de filas
- **Consolidación:** O(n log n) donde n = número de timestamps únicos
- **Detección de Señales:** O(n) donde n = número de snapshots
- **Simulación de Latencia:** O(m log n) donde m = señales, n = timestamps
- **Análisis:** O(m) donde m = número de señales

---

## Validaciones y Sanity Checks

### Validaciones de Datos

1. **Epoch:** Debe ser `int64`, no NaN, monotónico creciente
2. **Precios:** Positivos, no magic numbers, < MAX_REASONABLE_PRICE
3. **Spreads:** `px_ask >= px_bid` (no crossed books)
4. **Cantidades:** Positivas
5. **Book Identity:** QTE y STS deben pertenecer al mismo order book

### Sanity Checks de Oportunidades

1. **Spread Razonable:** < 50 bps
2. **Cantidades Realistas:** 100-10000 shares
3. **Venues Diferentes:** `max_bid_venue != min_ask_venue`
4. **Horario de Trading:** 09:00-17:30 CET

---

## Conclusión

Este sistema procesa datos de order books multi-venue para detectar oportunidades de arbitraje estadístico, simulando el impacto realista de latencia y generando métricas agregadas para análisis.

**Flujo Completo:**
1. Carga datos desde archivos CSV.GZ comprimidos
2. Limpia datos (magic numbers, trading status, validaciones)
3. Consolida datos multi-venue en un único tape temporal
4. Detecta oportunidades de arbitraje con rising edge detection
5. Simula impacto de latencia usando "Time Machine"
6. Genera métricas agregadas y reportes completos

**Características Clave:**
- Manejo robusto de datos faltantes y corruptos
- Validaciones exhaustivas en cada etapa
- Optimizaciones de rendimiento para datasets grandes
- Compatibilidad con múltiples venues y formatos de datos
- Simulación realista de latencia con búsqueda binaria eficiente

