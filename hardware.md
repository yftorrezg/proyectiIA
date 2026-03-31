# Hardware del sistema (ALTO RENDIMIENTO)

## CPU
- Intel Core i7-12700H
- 14 cores (6P + 8E)
- 20 threads
- Soporte excelente para multiprocessing y tareas paralelas

## GPU
- NVIDIA RTX 3060 Laptop (6GB VRAM)
- CUDA disponible
- Compatible con PyTorch, TensorFlow, OpenCV GPU

## RAM
- 32 GB DDR5

## Almacenamiento
- SSD 1TB

## Capacidades
- Procesamiento concurrente real
- Inferencia IA en GPU
- Streaming en tiempo real
- Manejo de múltiples pipelines simultáneos

## Objetivo del sistema
- Procesamiento en tiempo real (low latency)
- Uso híbrido CPU + GPU
- Pipeline paralelo (no secuencial)
- WebSockets sin bloqueo

## Reglas IMPORTANTES
- NO asumir CPU básica
- SIEMPRE considerar GPU cuando sea útil
- USAR multiprocessing o threading para tareas pesadas
- USAR asyncio para IO (WebSockets, streams)
- EVITAR loops bloqueantes