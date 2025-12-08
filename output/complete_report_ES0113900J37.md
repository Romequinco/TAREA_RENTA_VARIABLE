# REPORTE COMPLETO - ISIN: ES0113900J37

**Fecha de generación:** 2025-12-08 18:28:55

---

## RESUMEN EJECUTIVO

- **ISIN:** ES0113900J37
- **Total snapshots:** 307
- **Venues incluidos:** 4
- **Oportunidades detectadas:** 2
- **Profit teórico total:** €13.29
- **Oportunidades profitable:** 0
- **Profit real total:** €0.00

---

## TABLA COMPLETA DE TODAS LAS OPORTUNIDADES

```
           epoch venue_max_bid venue_min_ask  executable_qty  theoretical_profit  total_profit  bid_qty  ask_qty
1762523900000000          CEUX          XMAD           518.0        2.288818e-07      0.000119   3214.0    518.0
1762533000000000          AQEU          XMAD           949.0        1.400047e-02     13.286449    949.0 116559.0
```

## TABLA DE TRADES EJECUTADOS

## ANÁLISIS DE PARES DE VENUES

```
          Venue Pair  Count  Total Profit (€)  Avg Profit (€)  Max Profit (€)  Avg Price Diff (€)  Avg Qty
Buy@XMAD / Sell@AQEU      1         13.286449       13.286449       13.286449        1.400047e-02    949.0
Buy@XMAD / Sell@CEUX      1          0.000119        0.000119        0.000119        2.288818e-07    518.0
```

## MÉTRICAS DEL ANÁLISIS

```
                                                                                                                                                                                     detection                                                                  theoretical_profit                                                    real_profit                                                                                             temporal                                                                                                                                                                venue_pairs
{'total_snapshots': 307, 'snapshots_with_arbitrage': 2, 'arbitrage_rate_pct': 0.6514657980456027, 'total_rising_edges': 2, 'valid_opportunities': 2, 'detection_rate_pct': 0.6514657980456027} {'total': 13.286567459107028, 'mean': 6.643283729553514, 'max': 13.286448898315902} {'total': 0.0, 'profitable_count': 0, 'success_rate_pct': 0.0} {'start_time': 1970-01-21 09:35:23.900000, 'end_time': 1970-01-21 09:35:33, 'duration_seconds': 9.1}              Venue Pair  Count  Total Profit  Avg Profit
0  Buy@XMAD / Sell@AQEU      1     13.286449   13.286449
1  Buy@XMAD / Sell@CEUX      1      0.000119    0.000119
```

## GRÁFICAS GENERADAS

### Consolidated Tape

![Consolidated Tape](output\figures\consolidated_tape_ES0113900J37.png)

### Signals

![Signals](output\figures\signals_ES0113900J37.png)

### Latency Impact

![Latency Impact](output\figures\latency_impact_ES0113900J37.png)

## PASOS DEL PROCESAMIENTO

### Fase 1: Carga de Datos
- Carga de archivos QTE y STS
- Validación de columnas

### Fase 2: Limpieza
- Filtrado de magic numbers
- Validación de precios
- Filtrado por market status

### Fase 3: Consolidated Tape
- Merge multi-venue
- Forward fill

### Fase 4: Detección de Señales
- Cálculo de Global Max Bid / Min Ask
- Rising Edge Detection

### Fase 5: Simulación de Latencia
- Time Machine
- Re-cálculo de profit

### Fase 6: Análisis Final
- Análisis de oportunidades
- Estimación de ROI

---
*Documento generado automáticamente*