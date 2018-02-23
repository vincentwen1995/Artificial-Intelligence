package nl.tue.s2id90.group88;

import static java.lang.Integer.MAX_VALUE;
import static java.lang.Integer.MIN_VALUE;
import java.util.Collections;
import java.util.List;
import nl.tue.s2id90.draughts.DraughtsState;
import nl.tue.s2id90.draughts.player.DraughtsPlayer;
import org10x10.dam.game.Move;

/**
 * Implementation of the DraughtsPlayer interface.
 * @author huub
 */
// (DONE)ToDo: rename this class (and hence this file) to have a distinct name
//       for your player during the tournament
public class SithGuard_base  extends DraughtsPlayer{
    private int bestValue=0;
    int maxSearchDepth;
    
    /** boolean that indicates that the GUI asked the player to stop thinking. */
    private boolean stopped;
//    protected boolean playerColor;

    public SithGuard_base(int maxSearchDepth) {
        super("Emperor's_Shadow_Guard.jpeg"); // ToDo: replace with your own icon
        this.maxSearchDepth = maxSearchDepth;
    }
    
    @Override public Move getMove(DraughtsState s) {
        Move bestMove = null;
        bestValue = 0;
        DraughtsNode node = new DraughtsNode(s);    // the root of the search tree
//        playerColor = node.getState().isWhiteToMove();
        try {
                // compute bestMove and bestValue in a call to alphabeta
                maxSearchDepth = 0;
                while (true){
                    bestValue = alphaBeta(node, MIN_VALUE, MAX_VALUE, maxSearchDepth, 0);
                    bestMove  = node.getBestMove();
                    maxSearchDepth++;
                
                        // store the bestMove found uptill now
                        // NB this is not done in case of an AIStoppedException in alphaBeat()                
                    // print the results for debugging reasons
                    System.err.format(
                        "%s: depth= %2d, best move = %5s, value=%d\n", 
                        this.getClass().getSimpleName(),maxSearchDepth, bestMove, bestValue
                    );
                    }            
        } catch (AIStoppedException ex) {  /* nothing to do */  }
        
        if (bestMove==null) {
            System.err.println("no valid move found!");
            return getRandomValidMove(s);
        } else {
            return bestMove;
        }
    } 

    /** This method's return value is displayed in the AICompetition GUI.
     * 
     * @return the value for the draughts state s as it is computed in a call to getMove(s). 
     */
    @Override public Integer getValue() {
       return bestValue;
    }

    /** Tries to make alphabeta search stop. Search should be implemented such that it
     * throws an AIStoppedException when boolean stopped is set to true;
    **/
    @Override public void stop() {
       stopped = true; 
    }
    
    /** returns random valid move in state s, or null if no moves exist. */
    Move getRandomValidMove(DraughtsState s) {
        List<Move> moves = s.getMoves();
        Collections.shuffle(moves);
        return moves.isEmpty()? null : moves.get(0);
    }
    
    /** Implementation of alphabeta that automatically chooses the white player
     *  as maximizing player and the black player as minimizing player.
     * @param node contains DraughtsState and has field to which the best move can be assigned.
     * @param alpha
     * @param beta
     * @param depth maximum recursion Depth
     * @return the computed value of this node
     * @throws AIStoppedException
     **/
    int alphaBeta(DraughtsNode node, int alpha, int beta, int depth, int curDepth)
            throws AIStoppedException
    {
        if (node.getState().isWhiteToMove()) {
            return alphaBetaMax(node, alpha, beta, depth, curDepth);
        } else  {
            return alphaBetaMin(node, alpha, beta, depth, curDepth);
        }
    }
    
    /** Does an alphabeta computation with the given alpha and beta
     * where the player that is to move in node is the minimizing player.
     * 
     * <p>Typical pieces of code used in this method are:
     *     <ul> <li><code>DraughtsState state = node.getState()</code>.</li>
     *          <li><code> state.doMove(move); .... ; state.undoMove(move);</code></li>
     *          <li><code>node.setBestMove(bestMove);</code></li>
     *          <li><code>if(stopped) { stopped=false; throw new AIStoppedException(); }</code></li>
     *     </ul>
     * </p>
     * @param node contains DraughtsState and has field to which the best move can be assigned.
     * @param alpha
     * @param beta
     * @param depth  maximum recursion Depth
     * @return the compute value of this node
     * @throws AIStoppedException thrown whenever the boolean stopped has been set to true.
     */
     int alphaBetaMin(DraughtsNode node, int alpha, int beta, int depth, int curDepth)
            throws AIStoppedException {        
        if (stopped) { stopped = false; throw new AIStoppedException(); }
 //       System.out.println("Current Depth = " + Integer.toString(curDepth) + " alpha = " + Integer.toString(alpha) + " beta = " + Integer.toString(beta) );
        DraughtsState state = node.getState();
        if(curDepth >= depth){
//            System.out.println("Depth limit reached");
            return evaluate(node.getState());          
        }
        // ToDo: write an alphabeta search to compute bestMove and value        
        int bestVal = MAX_VALUE, i = 0;
        int bestMoveIndex = 0;
        int nextDepth;
        List<Move> moves = state.getMoves();        
        boolean isLeaf = moves.isEmpty();
        for (Move move : moves){           
            state.doMove(move);
            nextDepth = curDepth + 1;
            int value = alphaBetaMax(node, alpha, beta, depth, nextDepth);
            beta = Math.min(beta, value);            
            state.undoMove(move);
            if (beta <= alpha) {
                node.setBestMove(moves.get(bestMoveIndex));
                return alpha;
            }
            if (bestVal > value) {
                bestVal = value;
                bestMoveIndex = i;
            }    
            i++;
        }                
        if (isLeaf) {
            return evaluate(node.getState());
        }
        node.setBestMove(moves.get(bestMoveIndex));
        return bestVal;
     }
    
    int alphaBetaMax(DraughtsNode node, int alpha, int beta, int depth, int curDepth)
            throws AIStoppedException {
        if (stopped) { stopped = false; throw new AIStoppedException(); }
//        System.out.println("Current Depth = " + Integer.toString(curDepth) + " alpha = " + Integer.toString(alpha) + " beta = " + Integer.toString(beta) );
        DraughtsState state = node.getState();
        if(curDepth >= depth){
//            System.out.println("Depth limit reached");
            return evaluate(node.getState());
        }

        // (DONE)ToDo: write an alphabeta search to compute bestMove and value
        // ToDo: Implement Iterative Deepening
        int bestVal = MIN_VALUE, i = 0;
        int bestMoveIndex = 0;
        int nextDepth;
        List<Move> moves = state.getMoves();
        boolean isLeaf = moves.isEmpty();
        for (Move move : moves){
            state.doMove(move);            
            nextDepth = curDepth + 1;
            int value = alphaBetaMin(node, alpha, beta, depth, nextDepth);
            alpha = Math.max(alpha, value);            
            state.undoMove(move);
            if (alpha >= beta) {
                node.setBestMove(moves.get(bestMoveIndex));
                return beta;
            }
            if (bestVal < value){
                bestVal = value;
                bestMoveIndex = i;
            }
            i++;
        }
        if (isLeaf) {
            return evaluate(node.getState());
        }
        node.setBestMove(moves.get(bestMoveIndex));
        return bestVal;
    }

    /** A method that evaluates the given state. */
    // (DONE)ToDo: write an appropriate evaluation function
    int evaluate(DraughtsState state) { 
        int[] pieces = state.getPieces();
        int whiteCount = 0, blackCount = 0;
        for (int i = 1; i < 51; i++) {
            if (pieces[i] == DraughtsState.WHITEPIECE || pieces[i] == DraughtsState.WHITEKING){          //WHITEPIECE = 1, WHITEKING = 3
                whiteCount++;
            }
            else if(pieces[i] == DraughtsState.BLACKPIECE || pieces[i] == DraughtsState.BLACKKING) {      //BLACKPIECE = 2, BLACKKING = 4
                blackCount++;
            }
        }        
        return whiteCount - blackCount; 

    }
}
