"""
Módulo del Agente Minimax (Semana 2).
Implementa la toma de decisiones basada en árboles de juego y evaluación heurística.
"""


class MinimaxAgent:
    """
    Agente inteligente que utiliza el algoritmo Minimax con Backtracking
    para jugar de forma óptima al Tres en Raya.
    """

    def __init__(self, ai_player: str = 'O', human_player: str = 'X'):
        """
        Inicializa el agente con los roles de cada jugador y un contador de nodos.
        """
        self.ai_player = ai_player
        self.human_player = human_player
        
        # Contador para fines educativos: cuántos nodos del árbol se exploraron
        self.nodes_evaluated = 0

    def evaluate(self, board) -> int:
        """
        Función de evaluación heurística para estados terminales (Hojas del árbol).
        +10 si gana la IA, -10 si gana el humano, 0 para empate.
        """
        if board.check_winner(self.ai_player):
            return 10
        elif board.check_winner(self.human_player):
            return -10
        return 0

    def minimax(self, board, depth: int, is_maximizing: bool) -> int:
        """
        Algoritmo Minimax recursivo que explora el árbol de juego.
        
        Args:
            board: Estado actual del tablero (modelo).
            depth: Profundidad actual en el árbol (cuántos turnos adelante estamos simulando).
            is_maximizing: True si es el turno de la IA (MAX), False si es el turno del Humano (MIN).
            
        Returns:
            int: Puntuación óptima garantizada desde este nodo del árbol.
        """
        # Cada vez que entramos a esta función, visitamos un nodo del árbol
        self.nodes_evaluated += 1

        score = self.evaluate(board)

        # --- CASOS BASE: Nodos hoja del árbol de juego ---
        # Si la IA gana: restamos la profundidad para preferir victorias rápidas
        if score == 10:
            return score - depth
        
        # Si el humano gana: sumamos la profundidad para retrasar la derrota lo más posible
        if score == -10:
            return score + depth
        
        # Si ya no quedan casillas vacías, es un empate
        if not board.get_available_moves():
            return 0

        # --- PASO RECURSIVO ---
        if is_maximizing:
            # Turno de la IA (MAX): Quiere el número más grande posible
            best_score = -float('inf')  # Empezamos con -infinito

            for move in board.get_available_moves():
                # 1. Bajamos por la rama (hacemos la jugada simulada)
                board.make_move(move, self.ai_player)

                # 2. Llamada recursiva: ahora le toca al humano (is_maximizing=False)
                current_score = self.minimax(board, depth + 1, False)

                # 3. BACKTRACKING: borramos la jugada para volver a dejar el tablero como estaba
                board.undo_move(move)

                # 4. Nos quedamos con el puntaje más alto
                best_score = max(current_score, best_score)

            return best_score

        else:
            # Turno del Humano (MIN): La IA asume que el humano quiere el puntaje más bajo
            best_score = float('inf')  # Empezamos con +infinito

            for move in board.get_available_moves():
                # 1. Simulamos la jugada del humano
                board.make_move(move, self.human_player)

                # 2. Llamada recursiva: ahora le toca a la IA (is_maximizing=True)
                current_score = self.minimax(board, depth + 1, True)

                # 3. BACKTRACKING: borramos la jugada simulada
                board.undo_move(move)

                # 4. El humano elige el puntaje mínimo para perjudicar a la IA
                best_score = min(current_score, best_score)

            return best_score

    def get_best_move(self, board) -> tuple:
        """
        Evalúa todas las opciones actuales en el tablero real 
        y devuelve la mejor coordenada (fila, col) para la IA.
        """
        # Reiniciamos el contador de nodos antes de empezar a pensar
        self.nodes_evaluated = 0
        best_score = -float('inf')
        best_move = None

        # Probamos cada casilla vacía disponible
        for move in board.get_available_moves():
            # 1. Hacemos la jugada
            board.make_move(move, self.ai_player)

            # 2. Le preguntamos a Minimax qué pasaría si jugamos aquí
            # (el siguiente turno será del humano, por eso is_maximizing=False)
            score = self.minimax(board, 0, False)

            # 3. BACKTRACKING: deshacemos la jugada
            board.undo_move(move)

            # 4. Si esta jugada nos da un mejor resultado, la guardamos
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
