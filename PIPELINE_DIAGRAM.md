# Diagrama Completo del Pipeline de Arbitraje

## Visión General

Este documento describe el flujo completo del pipeline de análisis de arbitraje, incluyendo todas las funciones, módulos y pasos que se ejecutan.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARBITRAJE_PIPELINE.ipynb                     │
│                    (Notebook Principal)                         │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              │ Importa funciones desde módulos
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                       │
        ▼                     ▼                       ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ config_module│    │ data_loader_ │    │ data_cleaner_│
│              │    │ module       │    │ module       │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                       │
        │                     ▼                       │
        │            ┌──────────────┐                 │
        │            │consolidator_ │                 │
        │            │module        │                 │
        │            └──────────────┘                 │
        │                     │                       │
        │                     ▼                       │
        │            ┌──────────────┐                 │
        │            │signal_genera │                 │
        │            │tor_module    │                 │
        │            └──────────────┘                 │
        │                     │                       │
        │                     ▼                       │
        │            ┌──────────────┐                 │
        │            │latency_simula │                 │
        │            │tor_module     │                 │
        │            └──────────────┘                 │
        │                     │                       │
        │                     ▼                       │
        │            ┌──────────────┐                 │
        └───────────►│analyzer_module│◄───────────────┘
                    └──────────────┘
```

---

## Flujo Detallado del Pipeline

### PASO 1: CARGA DE DATOS

**Módulo:** `data_loader_module.py`

**Funciones utilizadas:**
- `find_all_isins(data_path, date)` - Descubre todos los ISINs únicos
- `load_data_for_isin(data_path, date, isin)` - Carga datos para un ISIN específico
  - Internamente llama a:
    - `load_qte_file(file_path)` - Carga archivo QTE (quotes)
    - `load_sts_file(file_path)` - Carga archivo STS (status)

**Proceso:**
```
1. Buscar archivos QTE y STS para cada exchange
   Patrón: QTE_YYYY-MM-DD_ISIN_TICKER_MIC_PART.csv.gz
   Patrón: STS_YYYY-MM-DD_ISIN_TICKER_MIC_PART.csv.gz

2. Para cada exchange:
   a. Leer archivo QTE comprimido (gzip)
      - Separador: ';'
      - Encoding: utf-8 o latin-1
      - Columnas requeridas: epoch, px_bid_0, px_ask_0, qty_bid_0, qty_ask_0
   
   b. Leer archivo STS comprimido (gzip)
      - Separador: ';'
      - Columnas requeridas: epoch, market_trading_status
      - Mapear market_trading_status → status

3. Retornar Dict[exchange] → (qte_df, sts_df)
```

**Output:**
- `data_dict`: Diccionario con estructura `{exchange: (qte_df, sts_df)}`

---

### PASO 2: FILTRADO Y LIMPIEZA

**Módulo:** `data_loader_module.py`

**Funciones utilizadas:**
- `is_valid_price(price)` - Verifica si un precio es válido
- `filter_valid_prices(df)` - Filtra filas con precios válidos
- `filter_continuous_trading(qte_df, sts_df, exchange)` - Filtra por estado de trading

**Proceso:**
```
1. FILTRO DE MAGIC NUMBERS
   Para cada fila en QTE:
   - Verificar que px_bid_0 no sea magic number
   - Verificar que px_ask_0 no sea magic number
   - Magic numbers: [666666.666, 999999.999, 999999.989, 999999.988, 999999.979, 999999.123]
   - Eliminar filas con magic numbers

2. FILTRO DE CONTINUOUS TRADING
   a. Hacer merge_asof de QTE con STS usando direction='backward'
      - Key: epoch
      - Propaga el último estado conocido a cada quote
   
   b. Filtrar por códigos de estado válidos:
      - BME: [5832713, 5832756]
      - AQUIS: [5308427]
      - CBOE: [12255233]
      - TURQUOISE: [7608181]
   
   c. Eliminar filas con estados inválidos

3. VALIDACIÓN DE PRECIOS
   Para cada fila:
   - px_bid_0 > 0
   - px_ask_0 > 0
   - px_bid_0 < px_ask_0 (no crossed)
   - qty_bid_0 > 0
   - qty_ask_0 > 0
   - Precios < 10000 EUR (sanity check)

4. Renombrar columnas:
   - px_bid_0 → bid
   - px_ask_0 → ask
   - qty_bid_0 → bidqty
   - qty_ask_0 → askqty
```

**Output:**
- `data_dict`: Diccionario con datos filtrados y limpios

---

### PASO 3: CONSOLIDACIÓN

**Módulo:** `consolidator_module.py`

**Funciones utilizadas:**
- `create_consolidated_tape(data_dict)` - Crea el tape consolidado

**Proceso:**
```
1. RENOMBRAR COLUMNAS POR EXCHANGE
   Para cada exchange en data_dict:
   - bid → {exchange}_bid
   - ask → {exchange}_ask
   - bidqty → {exchange}_bidqty
   - askqty → {exchange}_askqty

2. MERGE ITERATIVO
   a. Empezar con el primer exchange como base
   b. Para cada exchange adicional:
      - Hacer outer merge en 'epoch'
      - Mantener todos los timestamps únicos
   
   c. Resultado: DataFrame con todos los timestamps de todos los exchanges

3. ORDENAR POR EPOCH
   - sort_values('epoch')
   - Establecer 'epoch' como índice

4. FORWARD FILL
   - Asunción: El último precio conocido sigue vigente hasta el próximo update
   - ffill() para todas las columnas de precios
   - Previene NaNs en timestamps donde un exchange no actualiza

5. ELIMINAR FILAS INICIALES CON NaNs
   - Si todas las columnas son NaN, eliminar la fila
```

**Output:**
- `consolidated`: DataFrame con estructura:
  ```
  Index: epoch (int64)
  Columns: 
    - {exchange}_bid (float64)
    - {exchange}_ask (float64)
    - {exchange}_bidqty (float64)
    - {exchange}_askqty (float64)
    ... para cada exchange
  ```

---

### PASO 4: DETECCIÓN DE ARBITRAJE

**Módulo:** `signal_generator_module.py`

**Funciones utilizadas:**
- `detect_arbitrage_opportunities(consolidated)` - Detecta oportunidades

**Proceso:**
```
1. IDENTIFICAR COLUMNAS
   - bid_cols = [col for col in consolidated.columns if '_bid' in col and '_bidqty' not in col]
   - ask_cols = [col for col in consolidated.columns if '_ask' in col and '_askqty' not in col]

2. CALCULAR GLOBAL MAX BID
   Para cada timestamp:
   - global_max_bid = max(todos los bids disponibles)
   - venue_max_bid = exchange con el max_bid

3. CALCULAR GLOBAL MIN ASK
   Para cada timestamp:
   - global_min_ask = min(todos los asks disponibles)
   - venue_min_ask = exchange con el min_ask

4. APLICAR REGLA DE ARBITRAJE
   Para cada timestamp:
   - Si global_max_bid > global_min_ask:
     → Oportunidad de arbitraje detectada
     - profit_per_share = global_max_bid - global_min_ask
     - buy_exchange = venue_min_ask
     - sell_exchange = venue_max_bid
     - buy_price = global_min_ask
     - sell_price = global_max_bid
     - tradeable_qty = min(bid_qty de venue_max_bid, ask_qty de venue_min_ask)
     - total_profit = profit_per_share × tradeable_qty

5. Crear DataFrame con todas las oportunidades
```

**Output:**
- `opportunities`: DataFrame con columnas:
  - epoch
  - global_max_bid
  - global_min_ask
  - venue_max_bid
  - venue_min_ask
  - buy_exchange
  - sell_exchange
  - buy_price
  - sell_price
  - profit_per_share
  - tradeable_qty
  - total_profit

---

### PASO 5: RISING EDGE DETECTION

**Módulo:** `signal_generator_module.py`

**Funciones utilizadas:**
- `apply_rising_edge_detection(opportunities)` - Aplica filtro de liquidez

**Proceso:**
```
ALGORITMO: Rising Edge Detection v4 - Independent Side Tracking

1. INICIALIZACIÓN
   - last_bid_consumed = {}  # Dict[exchange] → última cantidad consumida en bid
   - last_ask_consumed = {}  # Dict[exchange] → última cantidad consumida en ask
   - is_rising_edge = []     # Lista de booleanos

2. PARA CADA OPORTUNIDAD (en orden cronológico):
   
   a. Obtener exchanges involucrados:
      - buy_exchange = venue_min_ask
      - sell_exchange = venue_max_bid
   
   b. TRACKING DE BID (lado de venta):
      - Si sell_exchange no está en last_bid_consumed:
        → Nueva oportunidad (rising edge)
        → last_bid_consumed[sell_exchange] = tradeable_qty
      
      - Si sell_exchange está en last_bid_consumed:
        - qty_available = bid_qty de sell_exchange en este timestamp
        - qty_consumed = last_bid_consumed[sell_exchange]
        - Si qty_available > qty_consumed:
          → Nueva oportunidad (rising edge)
          → last_bid_consumed[sell_exchange] = qty_consumed + tradeable_qty
        - Si qty_available <= qty_consumed:
          → Continuación de oportunidad anterior (no rising edge)
   
   c. TRACKING DE ASK (lado de compra):
      - Similar lógica para buy_exchange y ask_qty
   
   d. RISING EDGE = True solo si:
      - Hay nueva liquidez disponible en AMBOS lados (bid Y ask)
      - O es la primera oportunidad para ese par de exchanges

3. Agregar columna 'is_rising_edge' al DataFrame
```

**Output:**
- `opportunities_filtered`: DataFrame con columna adicional `is_rising_edge`
- Solo se consideran las oportunidades donde `is_rising_edge == True`

---

### PASO 6: SIMULACIÓN DE LATENCIA

**Módulo:** `latency_simulator_module.py`

**Funciones utilizadas:**
- `simulate_latency_with_losses(opportunities, consolidated, latency_us)` - Simula latencia

**Proceso:**
```
Para cada nivel de latencia (0µs, 100µs, 500µs, 1ms, 2ms, ..., 100ms):

1. PARA CADA OPORTUNIDAD:
   
   a. Obtener timestamp de detección: T = opportunity['epoch']
   
   b. Calcular timestamp de ejecución: T_exec = T + latency_us
   
   c. Obtener precios en T_exec usando get_quote_at_epoch():
      - buy_price_exec = get_quote_at_epoch(consolidated, buy_exchange, T_exec)
      - sell_price_exec = get_quote_at_epoch(consolidated, sell_exchange, T_exec)
   
   d. VALIDAR PRECIOS EN T_EXEC:
      - Si buy_price_exec es None o inválido → Oportunidad perdida
      - Si sell_price_exec es None o inválido → Oportunidad perdida
      - Si buy_price_exec >= sell_price_exec → Oportunidad perdida (spread cerrado)
   
   e. CALCULAR PROFIT/LOSS:
      - Si precios válidos:
        profit_per_share = sell_price_exec - buy_price_exec
        total_profit = profit_per_share × tradeable_qty
      - Si precios inválidos:
        total_profit = 0 (oportunidad perdida)
      - Si spread cerrado:
        total_profit = negativo (pérdida potencial)

2. SUMAR TODOS LOS PROFITS/LOSSES
   - total_profit = sum(total_profit de todas las oportunidades)

3. Retornar total_profit para este nivel de latencia
```

**Output:**
- `profits_by_latency`: Dict[latency_us] → total_profit
- Se almacena en `money_table_data`:
  ```python
  {
    'ISIN': isin,
    'Latency_us': latency_us,
    'Profit_EUR': total_profit
  }
  ```

---

### PASO 7: ANÁLISIS Y REPORTES

**Módulo:** `analyzer_module.py`

**Funciones utilizadas:**
- `create_money_table(money_table_data)` - Crea tabla pivot
- `create_decay_chart(money_table_data, save_path)` - Crea gráfico de decay
- `identify_top_opportunities(money_table_data, data_path, date, n)` - Top ISINs
- `generate_summary_answers(money_table_data)` - Resumen y respuestas

**Proceso:**

#### 7.1 Money Table
```
1. Convertir money_table_data a DataFrame
2. Crear tabla pivot:
   - Filas: ISINs
   - Columnas: Niveles de latencia
   - Valores: Profit_EUR (suma)
3. Agregar fila TOTAL
4. Crear tabla de resumen por latencia
```

#### 7.2 Decay Chart
```
1. Calcular totales por latencia
2. Crear gráfico con dos subplots:
   - Subplot 1: Profit vs Latency (lineal o log)
   - Subplot 2: % de profit respecto a latencia 0
3. Guardar en output/figures/decay_chart.png
```

#### 7.3 Top Opportunities
```
1. Filtrar ISINs con profit > 0 a latencia 0
2. Ordenar por profit descendente
3. Para cada top ISIN:
   - Recargar datos
   - Obtener estadísticas detalladas
   - Identificar mejor oportunidad individual
   - Sanity checks (verificar razonabilidad)
```

#### 7.4 Summary Answers
```
1. Calcular métricas clave:
   - max_profit (latencia 0)
   - profit_1ms, profit_10ms, profit_100ms
   - num_isins_with_opps
   - half_life_ms (vida media)
   - has_losses (si hay pérdidas)

2. Responder 3 preguntas:
   a. ¿Existen oportunidades de arbitraje?
   b. ¿Cuál es el profit máximo teórico?
   c. ¿Qué tan rápido desaparece el profit?
```

---

## Módulos y Funciones Principales

### config_module.py
- **Clase:** `Config`
- **Propósito:** Configuración centralizada
- **Parámetros clave:**
  - `DATA_SMALL_DIR`, `DATA_BIG_DIR`
  - `EXCHANGES`
  - `MAGIC_NUMBERS`
  - `CONTINUOUS_TRADING_STATUS`
  - `LATENCY_LEVELS`

### data_loader_module.py
- **Funciones:**
  - `is_valid_price(price)` → bool
  - `load_qte_file(file_path)` → DataFrame
  - `load_sts_file(file_path)` → DataFrame
  - `filter_valid_prices(df)` → DataFrame
  - `filter_continuous_trading(qte_df, sts_df, exchange)` → DataFrame
  - `load_data_for_isin(data_path, date, isin)` → Dict[exchange, (qte_df, sts_df)]
  - `find_all_isins(data_path, date)` → List[str]

### data_cleaner_module.py
- **Clase:** `DataCleaner`
- **Método:** `clean_all_venues(data_dict)` → Dict[exchange, qte_df]
- **Nota:** Wrapper que usa funciones de `data_loader_module`

### consolidator_module.py
- **Funciones:**
  - `create_consolidated_tape(data_dict)` → DataFrame
  - `get_quote_at_epoch(consolidated, exchange, epoch)` → Dict o None
- **Clase:** `ConsolidatedTape` (wrapper)

### signal_generator_module.py
- **Funciones:**
  - `detect_arbitrage_opportunities(consolidated)` → DataFrame
  - `apply_rising_edge_detection(opportunities)` → DataFrame
- **Clase:** `SignalGenerator` (wrapper)

### latency_simulator_module.py
- **Función:**
  - `simulate_latency_with_losses(opportunities, consolidated, latency_us)` → float

### analyzer_module.py
- **Funciones:**
  - `create_money_table(money_table_data)` → (pivot_df, summary_df)
  - `create_decay_chart(money_table_data, save_path)` → None
  - `identify_top_opportunities(money_table_data, data_path, date, n)` → DataFrame
  - `generate_summary_answers(money_table_data)` → Dict
- **Clase:** `ArbitrageAnalyzer` (wrapper)

---

## Flujo de Datos

```
Archivos CSV.gz
    ↓
load_qte_file() / load_sts_file()
    ↓
DataFrames raw (QTE, STS)
    ↓
filter_valid_prices() / filter_continuous_trading()
    ↓
DataFrames filtrados
    ↓
create_consolidated_tape()
    ↓
Consolidated DataFrame
    ↓
detect_arbitrage_opportunities()
    ↓
Opportunities DataFrame
    ↓
apply_rising_edge_detection()
    ↓
Filtered Opportunities DataFrame
    ↓
simulate_latency_with_losses() (para cada latencia)
    ↓
money_table_data (List[Dict])
    ↓
create_money_table() / create_decay_chart() / identify_top_opportunities() / generate_summary_answers()
    ↓
Reportes y Visualizaciones
```

---

## Dependencias entre Módulos

```
config_module
    ↑
    │ (importa)
    │
data_loader_module ──→ data_cleaner_module
    │
    │ (usa funciones)
    ↓
consolidator_module
    ↑
    │ (usa funciones)
    │
signal_generator_module
    ↑
    │ (usa funciones)
    │
latency_simulator_module
    ↑
    │ (usa funciones)
    │
analyzer_module
```

---

## Notas Importantes

1. **Filtrado:** El filtrado se aplica dentro de `load_data_for_isin()`, pero se muestra explícitamente en el pipeline como PASO 2.

2. **Forward Fill:** Crítico para la consolidación. Sin forward fill, no se pueden comparar precios entre exchanges.

3. **Rising Edge Detection:** Solo cuenta oportunidades únicas, evitando contar múltiples veces la misma oportunidad continua.

4. **Simulación de Latencia:** Usa `get_quote_at_epoch()` con búsqueda binaria (O(log n)) para eficiencia.

5. **Magic Numbers:** Valores especiales que NO son precios reales. Deben eliminarse siempre.

6. **Continuous Trading:** Solo se consideran válidos los períodos donde el mercado está en estado de trading continuo.

---

## Estructura de Archivos

```
TAREA_RENTA_VARIABLE/
├── ARBITRAJE_PIPELINE.ipynb    # Notebook principal
├── PIPELINE_DIAGRAM.md         # Este documento
├── src/
│   ├── config_module.py
│   ├── data_loader_module.py
│   ├── data_cleaner_module.py
│   ├── consolidator_module.py
│   ├── signal_generator_module.py
│   ├── latency_simulator_module.py
│   └── analyzer_module.py
├── data/
│   ├── DATA_SMALL/
│   └── DATA_BIG/
└── output/
    └── figures/
```

---

## Ejecución del Pipeline

1. **Configuración:** Cargar configuración desde `config_module`
2. **Descubrimiento:** Encontrar todos los ISINs disponibles
3. **Loop Principal:** Para cada ISIN:
   - Cargar datos
   - Filtrar y limpiar
   - Consolidar
   - Detectar oportunidades
   - Aplicar rising edge
   - Simular latencias
4. **Análisis Final:** Generar reportes y visualizaciones

---

*Última actualización: Diciembre 2025*

