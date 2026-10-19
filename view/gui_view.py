"""Vista (Tkinter): dibuja el tablero y el panel de control.

No contiene lógica del juego: avisa al controlador mediante callbacks.
"""

import tkinter as tk


class TresEnRayaView:
    """Ventana principal del juego."""

    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Tres en Raya: Minimax & Machine Learning (MVC)")
        self.ventana.geometry("560x390")
        self.ventana.resizable(False, False)

        # Funciones que registra el controlador para cada evento.
        self.click_callback = None
        self.modo_callback = None
        self.reiniciar_callback = None

        self.botones = [[None] * 3 for _ in range(3)]
        self._crear_interfaz()

    def _crear_interfaz(self):
        """Construye el tablero (izquierda) y el panel de control."""
        marco_tablero = tk.Frame(self.ventana)
        marco_tablero.pack(side=tk.LEFT, padx=20, pady=20)

        for fila in range(3):
            for col in range(3):
                boton = tk.Button(
                    marco_tablero,
                    text=" ",
                    width=5,
                    height=2,
                    font=("Arial", 20),
                    command=lambda f=fila, c=col: self._on_boton_click(f, c)
                )
                boton.grid(row=fila, column=col)
                self.botones[fila][col] = boton

        panel_control = tk.LabelFrame(
            self.ventana, text="Panel de Control", font=("Arial", 10))
        panel_control.pack(
            side=tk.RIGHT, padx=15, pady=15, fill=tk.BOTH, expand=True)

        # Mensaje de turno / resultado.
        self.etiqueta_mensaje = tk.Label(
            panel_control, text="Turno del Jugador: X",
            font=("Arial", 11, "bold"))
        self.etiqueta_mensaje.pack(pady=8, padx=5)

        # Selector de modo de juego.
        etiqueta_modo = tk.Label(
            panel_control, text="Modo de juego:", font=("Arial", 9, "bold"))
        etiqueta_modo.pack(anchor="w", padx=10, pady=(5, 0))

        self.modo_seleccionado = tk.StringVar(value="Humano vs Humano")
        opciones_modo = (
            ("Humano vs Humano", "Humano vs Humano"),
            ("Humano vs IA Minimax", "Humano vs IA Minimax"),
            ("Humano vs IA ML (Árbol)", "Humano vs IA Machine Learning"),
        )
        for texto, valor in opciones_modo:
            opcion = tk.Radiobutton(
                panel_control,
                text=texto,
                variable=self.modo_seleccionado,
                value=valor,
                command=self._on_modo_cambiado
            )
            opcion.pack(anchor="w", padx=10)

        # Métricas de la IA.
        marco_metricas = tk.LabelFrame(
            panel_control, text="Métricas de la IA", font=("Arial", 9, "bold"))
        marco_metricas.pack(fill=tk.X, padx=10, pady=8)

        self.etiqueta_nodos = tk.Label(
            marco_metricas, text="Nodos explorados: 0",
            font=("Arial", 9), anchor="w")
        self.etiqueta_nodos.pack(fill=tk.X, padx=5, pady=2)

        self.etiqueta_tiempo = tk.Label(
            marco_metricas, text="Tiempo: 0.0000 s",
            font=("Arial", 9), anchor="w")
        self.etiqueta_tiempo.pack(fill=tk.X, padx=5, pady=2)

        boton_reiniciar = tk.Button(
            panel_control,
            text="Reiniciar Partida",
            font=("Arial", 10),
            command=self._on_reiniciar_click
        )
        boton_reiniciar.pack(pady=8, padx=10)

    # --- Registro de callbacks (los asigna el controlador) ---

    def set_click_callback(self, callback):
        """Función a llamar con (fila, col) al pulsar una casilla."""
        self.click_callback = callback

    def set_modo_callback(self, callback):
        """Función a llamar con el nombre del modo elegido."""
        self.modo_callback = callback

    def set_reiniciar_callback(self, callback):
        """Función a llamar al pulsar 'Reiniciar Partida'."""
        self.reiniciar_callback = callback

    # --- Eventos internos: reenvían la acción al controlador ---

    def _on_boton_click(self, fila, col):
        if self.click_callback is not None:
            self.click_callback(fila, col)

    def _on_modo_cambiado(self):
        if self.modo_callback is not None:
            self.modo_callback(self.modo_seleccionado.get())

    def _on_reiniciar_click(self):
        if self.reiniciar_callback is not None:
            self.reiniciar_callback()

    # --- Métodos que el controlador usa para actualizar la pantalla ---

    def actualizar_tablero(self, tablero):
        """Muestra en los botones el contenido del tablero."""
        for fila in range(3):
            for col in range(3):
                self.botones[fila][col].config(text=tablero[fila][col])

    def actualizar_mensaje(self, mensaje):
        """Muestra el mensaje de turno o resultado."""
        self.etiqueta_mensaje.config(text=mensaje)

    def actualizar_metricas(self, nodos, tiempo_segundos):
        """Muestra los nodos explorados y el tiempo de respuesta de la IA."""
        self.etiqueta_nodos.config(text=f"Nodos explorados: {nodos:,}")
        self.etiqueta_tiempo.config(text=f"Tiempo: {tiempo_segundos:.4f} s")

    def refrescar(self):
        """Fuerza el redibujado (para ver 'pensando...' antes de calcular)."""
        self.ventana.update()
