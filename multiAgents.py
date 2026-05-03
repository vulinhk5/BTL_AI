# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # *** GIẢI THÍCH LOGIC BẰNG TIẾNG VIỆT ***
        # Hàm đánh giá này tính toán điểm số cho trạng thái tiếp theo sau khi thực hiện hành động.
        # Chúng ta sẽ xem xét 2 yếu tố chính: khoảng cách đến thức ăn gần nhất và khoảng cách đến ma.
        
        foodList = newFood.asList()
        
        # 1. Tìm khoảng cách ngắn nhất đến thức ăn.
        # Càng gần thức ăn, điểm đánh giá càng cao (sử dụng nghịch đảo khoảng cách: 1/khoảng_cách).
        minFoodDist = float('inf')
        for food in foodList:
            dist = util.manhattanDistance(newPos, food)
            if dist < minFoodDist:
                minFoodDist = dist
                
        if not foodList:
            minFoodDist = 0 # Trạng thái thắng (ăn hết thức ăn)
            
        # 2. Đánh giá sự nguy hiểm của các con ma.
        # Nếu có con ma nào không bị sợ hãi và ở quá gần (khoảng cách < 2), trạng thái này cực kỳ nguy hiểm.
        minGhostDist = float('inf')
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            dist = util.manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0: # Ma đang ở trạng thái săn đuổi
                if dist < minGhostDist:
                    minGhostDist = dist
        
        # Tránh xa trạng thái ma có thể ăn được pacman ngay lập tức
        if minGhostDist < 2:
            return float('-inf')
            
        # Điểm cơ bản được lấy từ điểm trò chơi (trạng thái tiếp theo)
        score = successorGameState.getScore()
        
        # Thưởng thêm nếu tiến gần đến thức ăn
        if minFoodDist > 0:
            score += 1.0 / minFoodDist
            
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # *** GIẢI THÍCH LOGIC BẰNG TIẾNG VIỆT ***
        # Minimax: Giả sử đối thủ (ma) luôn chọn nước đi gây bất lợi nhất cho Pacman (tối thiểu hóa điểm).
        # Pacman (tác nhân 0) sẽ cố gắng chọn nước đi có điểm lớn nhất trong các trường hợp tồi tệ nhất đó.
        
        def minimax(agentIndex, depth, gameState):
            # Điều kiện dừng: trạng thái thắng, thua hoặc đã duyệt tới độ sâu giới hạn.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Nếu là Pacman (chọn Max)
            if agentIndex == 0:
                v = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    v = max(v, minimax(1, depth, gameState.generateSuccessor(agentIndex, action)))
                return v
            
            # Nếu là Ma (chọn Min)
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                # Nếu tất cả các ma đã đi xong ở độ sâu hiện tại, chuyển lại cho Pacman và tăng độ sâu.
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    nextDepth += 1
                
                v = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    v = min(v, minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)))
                return v

        # Logic gọi ở gốc cây quyết định (trạng thái hiện tại của Pacman)
        bestAction = Directions.STOP
        bestScore = float('-inf')
        
        for action in gameState.getLegalActions(0):
            # Gọi hàm minimax cho các tác nhân Ma, bắt đầu từ Ma số 1
            score = minimax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
                
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # *** GIẢI THÍCH LOGIC BẰNG TIẾNG VIỆT ***
        # Alpha-Beta Pruning tối ưu hóa thuật toán Minimax bằng cách cắt tỉa (bỏ qua)
        # các nhánh không bao giờ thay đổi được kết quả cuối cùng.
        # Alpha: giá trị tốt nhất hiện tại (cao nhất) cho ngã rẽ ở đường dẫn từ gốc tới lá của Max
        # Beta: giá trị tốt nhất hiện tại (thấp nhất) cho ngã rẽ ở đường dẫn từ gốc tới lá của Min
        
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            # Điều kiện dừng
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
                
            # Pacman (Max Node)
            if agentIndex == 0:
                v = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    v = max(v, alphabeta(1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    # Nếu điểm v tìm được lớn hơn giá trị giới hạn beta hiện tại, trả về v (cắt tỉa)
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
                
            # Ghosts (Min Node)
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    nextDepth += 1
                    
                v = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    v = min(v, alphabeta(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    # Nếu v nhỏ hơn giá trị giới hạn alpha hiện tại, trả về v (cắt tỉa)
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        bestAction = Directions.STOP
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            score = alphabeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
            
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # *** GIẢI THÍCH LOGIC BẰNG TIẾNG VIỆT ***
        # Expectimax xử lý đối thủ không tối ưu bằng cách lấy giá trị trung bình kì vọng.
        # Ở nhánh của Ma, thay vì chọn nước đi làm giảm điểm nhiều nhất (như Minimax),
        # ta tính trung bình cộng điểm số của tất cả các nước đi khả thi (vì ma có thể di chuyển ngẫu nhiên).
        
        def expectimax(agentIndex, depth, gameState):
            # Điều kiện dừng
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
                
            # Pacman (Max Node)
            if agentIndex == 0:
                v = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    v = max(v, expectimax(1, depth, gameState.generateSuccessor(agentIndex, action)))
                return v
                
            # Ghosts (Expectation Node)
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    nextDepth += 1
                    
                actions = gameState.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return 0
                
                # Tính trung bình điểm kì vọng (giả định xác suất các hành động của ma là như nhau)
                totalScore = 0
                for action in actions:
                    totalScore += expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                    
                return float(totalScore) / len(actions)

        bestAction = Directions.STOP
        bestScore = float('-inf')
        
        for action in gameState.getLegalActions(0):
            score = expectimax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
                
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    1. Đánh giá khoảng cách gần nhất đến thức ăn: Khoảng cách càng nhỏ càng có lợi.
    2. Đánh giá mối đe dọa của ma: Nếu ma không bị sợ hãi và ở quá gần (khoảng cách <= 1), trừ điểm rất nặng. 
       Nếu ma đang sợ hãi, tiến lại gần nó sẽ được thưởng thêm điểm.
    3. Đánh giá viên sức mạnh (capsules): Số lượng capsules còn lại càng ít thì càng tốt.
    """
    # *** GIẢI THÍCH LOGIC BẰNG TIẾNG VIỆT ***
    # Đây là hàm đánh giá trạng thái "xịn" hơn. Thay vì chỉ sử dụng khi thử hành động,
    # hàm này sẽ đánh giá tổng thể độ "tốt" của một trạng thái tĩnh hiện tại trong bản đồ.
    
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    
    score = currentGameState.getScore()
    
    # Ưu tiên đi ăn các viên thức ăn gần nhất
    minFoodDist = float('inf')
    for food in foodList:
        dist = util.manhattanDistance(pos, food)
        if dist < minFoodDist:
            minFoodDist = dist
            
    if minFoodDist < float('inf'):
        score += 1.0 / minFoodDist
        
    # Tránh ma thường và khuyến khích ăn ma đang hoảng sợ
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = util.manhattanDistance(pos, ghostPos)
        if ghostState.scaredTimer > 0: 
            # Ma đang bị sợ hãi, khoảng cách càng gần phần thưởng càng cao để ưu tiên đi ăn
            if dist > 0:
                score += 2.0 / dist
        else:
            # Ma không sợ hãi, nếu quá gần thì đây là trạng thái cực kỳ tệ, trừ điểm rất nhiều
            if dist <= 1:
                score -= 1000
                
    # Ưu tiên ăn capsules. Càng nhiều capsule chưa ăn, điểm càng bị trừ nhiều
    score -= len(capsules) * 10
    
    return score

# Abbreviation
better = betterEvaluationFunction

def riskAwareEvaluationFunction(currentGameState: GameState):
    """
    Một hàm đánh giá (Evaluation Function) cân nhắc rủi ro, 
    đặc biệt chú trọng đến việc sống sót và giữ khoảng cách an toàn.
    """
    # *** GIẢI THÍCH LOGIC (RISK-AWARE) BẰNG TIẾNG VIỆT ***
    # Trọng tâm của hàm này là ĐÁNH GIÁ RỦI RO (Risk). 
    # Thay vì cực kỳ tham ăn như hàm 'better', hàm này ưu tiên việc giữ mạng sống.
    
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    
    # 1. Tính điểm cơ bản
    score = currentGameState.getScore()
    
    # 2. Tính toán mối đe dọa từ ma (Yếu tố Rủi Ro cao nhất)
    minGhostDist = float('inf')
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = util.manhattanDistance(pos, ghostPos)
        
        # Nếu ma không sợ hãi (đang săn đuổi)
        if ghostState.scaredTimer == 0:
            if dist < minGhostDist:
                minGhostDist = dist
                
            # Hình phạt rủi ro: 
            # - Nếu ma quá gần (khoảng cách <= 2), rủi ro tử vong là CỰC KỲ CAO.
            #   Ta trừ điểm rất nặng để AI tuyệt đối tránh né.
            if dist <= 2:
                score -= 5000 
            # - Nếu ma ở khoảng cách trung bình (3 đến 5), rủi ro đang tăng lên.
            #   Ta trừ một lượng điểm tỷ lệ nghịch với khoảng cách. 
            #   (Càng gần thì trừ càng nhiều).
            elif dist <= 5:
                score -= 100.0 / dist
                
        # Nếu ma đang sợ hãi (Pacman có thể ăn ma)
        elif ghostState.scaredTimer > 0:
            # Chỉ khuyến khích đi ăn ma nếu thời gian sợ hãi còn đủ dài (rủi ro thấp).
            # Nếu thời gian sắp hết, không nên mạo hiểm đuổi theo.
            if ghostState.scaredTimer > dist / 2: 
                score += 50.0 / dist

    # 3. Tính toán khoảng cách đến thức ăn (Yếu tố Phần Thưởng)
    minFoodDist = float('inf')
    for food in foodList:
        dist = util.manhattanDistance(pos, food)
        if dist < minFoodDist:
            minFoodDist = dist
            
    # Thưởng điểm khi lại gần thức ăn, nhưng phần thưởng này không bao giờ 
    # được phép lớn hơn hình phạt rủi ro khi ma ở gần.
    if minFoodDist < float('inf'):
        score += 10.0 / minFoodDist

    return score

