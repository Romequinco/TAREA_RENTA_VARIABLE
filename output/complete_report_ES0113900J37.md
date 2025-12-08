# REPORTE COMPLETO - ISIN: ES0113900J37

**Fecha de generación:** 2025-12-08 17:13:42

---

## RESUMEN EJECUTIVO

- **ISIN:** ES0113900J37
- **Total snapshots:** 307
- **Venues incluidos:** 4
- **Oportunidades detectadas:** 0
- **Profit teórico total:** €0.00

---

## TABLA COMPLETA DE TODAS LAS OPORTUNIDADES

## MÉTRICAS DEL ANÁLISIS

```
                                                                                                                                                       detection theoretical_profit real_profit temporal venue_pairs spreads
{'total_snapshots': 307, 'snapshots_with_arbitrage': 0, 'arbitrage_rate_pct': 0.0, 'total_rising_edges': 0, 'valid_opportunities': 0, 'detection_rate_pct': 0.0}               None        None     None        None    None
```

## GRÁFICAS GENERADAS

### Consolidated Tape

![Consolidated Tape](output\figures\consolidated_tape_ES0113900J37.png)

### Signals

![Signals](output\figures\signals_ES0113900J37.png)

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