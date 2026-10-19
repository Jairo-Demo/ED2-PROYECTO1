import tkinter as tk


class TresEnRayaView:
    def __init__(self, root):
        # Guardo la ventana principal
        self.root = root
        self.root.title("Tres en Raya con Minimax (IA)")
        self.root.geometry("520x360")
        self.root.resizable(False, False)
        
        # Callbacks para avisarle al controlador cuando ocurren eventos
        self.click_callback = None
        self.modo_callback = None
        self.reiniciar_callback = None
        
        # Creo una matriz 3x3 a mano para guardar mis botones
        self.botones = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]
        
        # Llamo a la función que dibuja la pantalla
        self._crear_interfaz()

    def _crear_interfaz(self):
        # --- PANEL IZQUIERDO: Tablero ---
        frame_tablero = tk.Frame(self.root)
        frame_tablero.pack(side=tk.LEFT, padx=20, pady=20)

        # Creamos los 9 botones de la matriz
        for fila in range(3):
            for col in range(3):
                boton = tk.Button(
                    frame_tablero, 
                    text=" ", 
                    width=5, 
                    height=2,
                    font=("Arial", 20),
                    command=lambda f=fila, c=col: self._on_boton_click(f, c)
                )
                boton.grid(row=fila, column=col)
                self.botones[fila][col] = boton

        # --- PANEL LATERAL DE CONTROL Y MÉTRICAS ---
        panel_lateral = tk.LabelFrame(self.root, text="Panel de Control", font=("Arial", 10))
        panel_lateral.pack(side=tk.RIGHT, padx=15, pady=15, fill=tk.BOTH, expand=True)

        # 1. Estado del juego
        self.lbl_mensaje = tk.Label(panel_lateral, text="Turno del Jugador: X", font=("Arial", 11, "bold"))
        self.lbl_mensaje.pack(pady=8, padx=5)

        # 2. Selector de modo de juego (Semana 2 - Día 4)
        lbl_modo = tk.Label(panel_lateral, text="Modo de juego:", font=("Arial", 9, "bold"))
        lbl_modo.pack(anchor="w", padx=10, pady=(5, 0))

        self.var_modo = tk.StringVar(value="Humano vs Humano")

        rb1 = tk.Radiobutton(
            panel_lateral, 
            text="Humano vs Humano", 
            variable=self.var_modo, 
            value="Humano vs Humano",
            command=self._on_modo_cambiado
        )
        rb1.pack(anchor="w", padx=10)

        rb2 = tk.Radiobutton(
            panel_lateral, 
            text="Humano vs IA Minimax", 
            variable=self.var_modo, 
            value="Humano vs IA Minimax",
            command=self._on_modo_cambiado
        )
        rb2.pack(anchor="w", padx=10)

        # 3. Métricas de Rendimiento (Semana 2 - Día 5)
        frame_metricas = tk.LabelFrame(panel_lateral, text="Métricas de la IA", font=("Arial", 9, "bold"))
        frame_metricas.pack(fill=tk.X, padx=10, pady=8)

        self.lbl_nodos = tk.Label(frame_metricas, text="Nodos explorados: 0", font=("Arial", 9), anchor="w")
        self.lbl_nodos.pack(fill=tk.X, padx=5, pady=2)

        self.lbl_tiempo = tk.Label(frame_metricas, text="Tiempo: 0.0000 s", font=("Arial", 9), anchor="w")
        self.lbl_tiempo.pack(fill=tk.X, padx=5, pady=2)

        # 4. Botón para reiniciar partida
        btn_reiniciar = tk.Button(
            panel_lateral, 
            text="Reiniciar Partida", 
            font=("Arial", 10),
            command=self._on_reiniciar_click
        )
        btn_reiniciar.pack(pady=8, padx=10)

    # Métodos de configuración de callbacks (MVC)
    def set_click_callback(self, callback):
        self.click_callback = callback

    def set_modo_callback(self, callback):
        self.modo_callback = callback

    def set_reiniciar_callback(self, callback):
        self.reiniciar_callback = callback

    # Manejadores de eventos internos
    def _on_boton_click(self, fila, col):
        if self.click_callback is not None:
            self.click_callback(fila, col)

    def _on_modo_cambiado(self):
        if self.modo_callback is not None:
            self.modo_callback(self.var_modo.get())

    def _on_reiniciar_click(self):
        if self.reiniciar_callback is not None:
            self.reiniciar_callback()

    # Métodos para actualizar la pantalla desde el controlador
    def actualizar_tablero(self, tablero_modelo):
        for fila in range(3):
            for col in range(3):
                valor = tablero_modelo[fila][col]
                self.botones[fila][col].config(text=valor)

    def actualizar_mensaje(self, mensaje):
        self.lbl_mensaje.config(text=mensaje)

    def actualizar_metricas(self, nodos: int, tiempo_segundos: float):
        """Actualiza en tiempo real los nodos explorados y la latencia de la IA."""
        self.lbl_nodos.config(text=f"Nodos explorados: {nodos:,}")
        self.lbl_tiempo.config(text=f"Tiempo: {tiempo_segundos:.4f} s")
