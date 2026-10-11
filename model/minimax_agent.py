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
        
        Args:
            ai_player (str): Ficha utilizada por la IA (por defecto 'O').
            human_player (str): Ficha utilizada por el jugador humano (por defecto 'X').
        """
        self.ai_player = ai_player
        self.human_player = human_player
        
        # Contador para fines educativos: cuántos nodos del árbol se revisaron
        self.nodes_evaluated = 0

    def evaluate(self, board) -> int:
        """
        Función de evaluación heurística para estados terminales (Hojas del árbol).
        
        Asigna una puntuación numérica al estado del tablero:
            +10: Victoria de la IA (el mejor escenario posible para la máquina).
            -10: Victoria del Humano (el peor escenario posible para la máquina).
              0: Empate o juego en curso (escenario neutral).
              
        Args:
            board (TresEnRayaModel): Estado actual del tablero.
            
        Returns:
            int: Puntuación calculada (+10, -10 o 0).
        """
        # Caso 1: La IA logró tres en raya
        if board.check_winner(self.ai_player):
            return 10
        
        # Caso 2: El humano logró tres en raya
        elif board.check_winner(self.human_player):
            return -10
        
        # Caso 3: Empate o no hay ganador
        return 0
