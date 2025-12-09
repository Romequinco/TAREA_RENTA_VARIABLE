# TAREA_RENTA_VARIABLE

Tarea n.1 del Bloque de Renta Variable del grado mIAx

## Estructura del Proyecto

Este proyecto implementa un pipeline completo para detectar y analizar oportunidades de arbitraje en mercados de renta variable europeos.

### Módulos en `src/`

Todos los módulos están organizados en el directorio `src/` y contienen funciones especializadas:

- **config_module.py**: Configuración del sistema (paths, exchanges, magic numbers, etc.)
- **data_loader_module.py**: Carga y filtrado de archivos QTE (quotes) y STS (status)
- **data_cleaner_module.py**: Limpieza y validación de datos
- **consolidator_module.py**: Creación del consolidated tape fusionando datos de múltiples exchanges
- **signal_generator_module.py**: Detección de oportunidades de arbitraje
- **latency_simulator_module.py**: Simulación del impacto de la latencia en la ejecución
- **analyzer_module.py**: Análisis y visualización de resultados

### Notebook Principal

**ARBITRAJE_PIPELINE.ipynb**: Notebook completo que contiene:

- Explicaciones detalladas de cada fase del pipeline
- Código ejecutable con todas las funciones implementadas
- Análisis de resultados y visualizaciones
- Documentación integrada en formato markdown

Todo el código está explicado y ejecutado directamente en el notebook, que sirve como documentación interactiva del sistema completo.
