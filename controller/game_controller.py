class GameController:
    def __init__(self, modelo, vista):
        # Guardamos el modelo (reglas) y la vista (interfaz)
        self.modelo = modelo
        self.vista = vista
        
        # El juego siempre empieza con la ficha X
        self.turno_actual = 'X'
        
        # Aquí conectamos la interfaz con las reglas:
        # Le decimos a la vista "cuando hagan clic, llama a esta función"
        self.vista.set_click_callback(self.manejar_clic)

    def manejar_clic(self, fila, col):
        # 1. Le pedimos al modelo que intente poner la ficha en esa fila y columna
        movimiento_exitoso = self.modelo.hacer_movimiento(fila, col, self.turno_actual)
        
        # Si la casilla estaba vacía, el movimiento es exitoso (True)
        if movimiento_exitoso:
            # 2. Copiamos los datos de la matriz del modelo a los botones de la vista
            self.vista.actualizar_tablero(self.modelo.tablero)
            
            # 3. Revisamos si con este movimiento alguien ganó
            if self.modelo.verificar_victoria(self.turno_actual):
                self.vista.actualizar_mensaje(f"¡GANÓ EL JUGADOR {self.turno_actual}!")
                # Desactivamos los clics para que no sigan jugando
                self.vista.set_click_callback(None)
            
            # 4. Revisamos si el tablero se llenó y es un empate
            elif self.modelo.es_empate():
                self.vista.actualizar_mensaje("¡EMPATE! Tablero lleno.")
            
            # 5. Si no hay ganador ni empate, cambiamos de turno
            else:
                if self.turno_actual == 'X':
                    self.turno_actual = 'O'
                else:
                    self.turno_actual = 'X'
                
                # Le pedimos a la vista que cambie el panel lateral para mostrar el nuevo turno
                self.vista.actualizar_mensaje(f"Turno del Jugador: {self.turno_actual}")
