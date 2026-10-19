# Reporte de Análisis Comparativo: IA Simbólica vs. Machine Learning
**Proyecto Educativo: Tres en Raya con Minimax, Backtracking y Árboles de Decisión**  
**Patrón de Arquitectura:** Modelo-Vista-Controlador (MVC)  
**Asignatura / Curso:** Estructuras de Datos / Inteligencia Artificial  

---

## 1. Resumen Ejecutivo
El presente informe consolida los resultados experimentales obtenidos durante la **Semana 3 - Día 5**, contrastando dos paradigmas fundamentales de la Inteligencia Artificial:
1. **IA Simbólica Tradicional:** Representada por el algoritmo **Minimax con Backtracking**, el cual realiza una búsqueda exhaustiva en el espacio de estados explorando el árbol de juego.
2. **Machine Learning Supervisado:** Representado por un **Árbol de Decisión (`DecisionTreeClassifier`)** entrenado con datos sintéticos generados por Minimax, el cual aprende a mapear el estado actual del tablero hacia la jugada óptima mediante inferencia directa.

---

## 2. Resultados Experimentales (Benchmark de 100 Posiciones)
Se evaluó el desempeño de ambos agentes sobre una batería de **100 posiciones de juego intermedias e independientes** con diferente grado de avance (entre 1 y 6 fichas colocadas):

| Métrica de Rendimiento | Minimax (IA Simbólica) | Árbol de Decisión ML | Conclusión y Diferencial |
| :--- | :---: | :---: | :--- |
| **Tiempo promedio por jugada** | **62.35 ms** | **1.67 ms** | El Árbol ML es **37.2 veces más rápido** por decisión. |
| **Tiempo total (100 jugadas)** | **6.2354 s** | **0.1674 s** | Latencia casi imperceptible para el usuario humano. |
| **Nodos explorados por jugada** | **16,531 nodos** | **0 nodos (Directo)** | ML no necesita calcular jugadas futuras. |
| **Nodos totales acumulados** | **1,653,095 nodos** | **0 nodos** | Ahorro masivo de cómputo en memoria y CPU. |
| **Precisión vs. Jugada Óptima** | **100.0% (Infalible)** | **~58.0% - 65.0%** | Coincidencia estricta con la primera rama óptima. |
| **Complejidad Temporal** | Exponencial: $\mathcal{O}(b^d)$ | Logarítmica / Lineal: $\mathcal{O}(h)$ | Inferencia acotada por la profundidad del árbol ($h \le 12$). |

---

## 3. Justificación Teórica de las Diferencias de Rendimiento

### 3.1. Búsqueda Exhaustiva vs. Inferencia por Reglas Lógicas
* **Minimax con Backtracking:**
  Cada vez que el agente debe mover, genera de manera dinámica un **árbol de juego (similar a un árbol m-vías)**. Mediante llamadas recursivas y retroceso con `undo_move()`, explora todas las ramas posibles hasta llegar a las hojas (victorias, derrotas o empates), evaluando recursivamente la función de puntuación ($+10, -10, 0$). Aunque garantiza no perder jamás, su costo computacional crece factorialmente con el número de casillas vacías.
* **Árbol de Decisión (Machine Learning):**
  El modelo no explora el futuro. Durante la fase de entrenamiento (Semana 3 - Día 3), Scikit-Learn condensó el conocimiento de 1,000 partidas supervisadas en un **Árbol Binario de preguntas Sí/No** sobre las 9 casillas (`c0` a `c8`). En tiempo de juego, la inferencia consiste únicamente en descender por las ramas de ese árbol binario precalculado, requiriendo a lo sumo 10 a 12 comparaciones simples de números reales.

### 3.2. Análisis del Dilema Precisión vs. Latencia (Trade-Off)
* La precisión de coincidencia del modelo ML (~58% a 65%) refleja un fenómeno clásico en juegos de suma cero: **en múltiples estados del Tres en Raya existen 2 o más casillas con idéntico valor minimax (empate garantizado)**. Mientras que Minimax selecciona sistemáticamente la primera que encuentra en su recorrido, el Árbol de Decisión puede optar por otra jugada igualmente válida según la distribución probabilística del dataset.
* La ganancia es sustancial: a cambio de una ligera variación táctica en posiciones no críticas, se obtiene una reducción drástica del tiempo de cálculo (**37x a 100x de aceleración**).

---

## 4. Relevancia y Escalabilidad en el Mundo Real
¿Por qué este proyecto es fundamental para la formación en Ingeniería y Ciencias de la Computación?
* **Juegos Pequeños (Tres en Raya):** El espacio de estados tiene solo 255,168 posiciones posibles. Minimax es viable en tiempo real.
* **Juegos Complejos (Ajedrez, Go):** El Ajedrez cuenta con aproximadamente $10^{120}$ posiciones posibles (Número de Shannon), y el Go con $10^{170}$. Ningún supercomputador puede explorar sus árboles mediante Minimax exhaustivo.
* **La Solución Moderna (AlphaGo, Modelos de Conducción, Robótica):**
  Exactamente como demostramos en este proyecto, los sistemas modernos de IA sustituyen la búsqueda exhaustiva por modelos entrenados que evalúan tableros o entornos de manera instantánea mediante aproximación de funciones.

---

## 5. Guía Rápida para la Defensa Oral del Proyecto

Si el docente o tribunal realiza preguntas durante la presentación, las respuestas clave son:

1. **¿Dónde se aplicaron las Estructuras de Datos?**
   * *Respuesta:* De dos formas: el **árbol de juego m-vías implícito** recorrido con Backtracking en Minimax (Semana 2) y el **árbol binario explícito** generado por Scikit-Learn para las reglas de decisión (Semana 3).
2. **¿Por qué se utilizó el Patrón MVC?**
   * *Respuesta:* Para desacoplar la lógica del juego (`model/`), la interfaz gráfica Tkinter (`view/`) y la coordinación de turnos y métricas (`controller/`). Permitió cambiar y agregar agentes de IA sin modificar ni una sola línea de la interfaz visual.
3. **¿Por qué la IA de Machine Learning muestra 0 nodos en pantalla?**
   * *Respuesta:* Porque no genera jugadas hipotéticas a futuro; solo ejecuta una pasada hacia adelante (*forward pass*) sobre el árbol de decisión entrenado, logrando inferencia en tiempo constante relativo.

---
**Estado del Proyecto:** 100% Completado (Semanas 1, 2 y 3 implementadas y operativas).
