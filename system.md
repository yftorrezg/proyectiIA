  El sistema corre en hardware de alto rendimiento:

- CPU multinúcleo (20 threads)
- GPU NVIDIA RTX con CUDA
- 32GB RAM

Por lo tanto:

- usa paralelismo real
- usa asyncio para IO
- usa multiprocessing para CPU-bound
- usa GPU para visión e inferencia
- diseña pipelines concurrentes

Evita:
- código secuencial bloqueante
- soluciones simplistas