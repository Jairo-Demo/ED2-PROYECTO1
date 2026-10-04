import tkinter as tk

class TresEnRayaView:
    def __init__(self, root):
        # Guardo la ventana principal
        self.root = root
        self.root.title("Tres en Raya")
        
        # Esta variable guardará la función que el controlador me pase
        # para avisarle cuando el usuario haga clic en un botón.
        self.click_callback = None
        
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

        # Hago 2 ciclos for para crear los 9 botones de la matriz
        for fila in range(3):
            for col in range(3):
                # Creo un botón. 
                # Uso lambda para atrapar la fila y columna en la que estoy creando este botón.
                boton = tk.Button(
                    frame_tablero, 
                    text=" ", 
                    width=5, 
                    height=2,
                    font=("Arial", 20),
                    command=lambda f=fila, c=col: self._on_boton_click(f, c)
                )
                # Lo ubico en la grilla visual
                boton.grid(row=fila, column=col)
                # Guardo el botón en mi matriz de memoria
                self.botones[fila][col] = boton

        # --- PANEL LATERAL DE ESTADO ---
        # LabelFrame crea un cuadro visible con un título arriba
        panel_lateral = tk.LabelFrame(self.root, text="Panel de Estado", font=("Arial", 10))
        panel_lateral.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Etiqueta adentro del panel que mostrará el mensaje ("Turno de X", etc)
        self.lbl_mensaje = tk.Label(panel_lateral, text="Turno del Jugador: X", font=("Arial", 12))
        self.lbl_mensaje.pack(pady=20, padx=10)

    def set_click_callback(self, callback):
        # El controlador usa este método para "enchufar" su función a esta vista
        self.click_callback = callback

    def _on_boton_click(self, fila, col):
        # Si el usuario hace clic, le aviso al controlador mandándole la fila y columna
        if self.click_callback != None:
            self.click_callback(fila, col)

    def actualizar_tablero(self, tablero_modelo):
        # Recorro todo el tablero del modelo (que tiene 'X' o 'O') 
        # y copio esos valores a mis botones visuales
        for fila in range(3):
            for col in range(3):
                valor = tablero_modelo[fila][col]
                self.botones[fila][col].config(text=valor)

    def actualizar_mensaje(self, mensaje):
        # Cambio el texto de la etiqueta lateral
        self.lbl_mensaje.config(text=mensaje)
