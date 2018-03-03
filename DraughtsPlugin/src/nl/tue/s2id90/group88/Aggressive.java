package nl.tue.s2id90.group88;

import static java.lang.Integer.MAX_VALUE;
import static java.lang.Integer.MIN_VALUE;
import java.util.Collections;
import java.util.List;
//import java.util.ArrayList;
import nl.tue.s2id90.draughts.DraughtsState;
import nl.tue.s2id90.draughts.player.DraughtsPlayer;
import org10x10.dam.game.Move;

/**
 * Implementation of the DraughtsPlayer interface.
 * @author huub
 */
// (DONE)ToDo: rename this class (and hence this file) to have a distinct name
//       for your player during the tournament
public class Aggressive  extends DraughtsPlayer{
    private int bestValue=0;
    int maxSearchDepth;
    public int whitePiece = 20;
    public int whiteKing = 0;
    public int blackPiece = 20;
    public int blackKing = 0;
    public int whiteCount = 20;
    public int blackCount = 20;
    public int lastwhiteCount = 20;
    public int lastblackCount = 20;
    /** boolean that indicates that the GUI asked the player to stop thinking. */
    private boolean stopped;
//    protected boolean playerColor;

    public Aggressive(int maxSearchDepth) {
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
                    maxSearchDepth = maxSearchDepth + 2;    //Focusing on optimizing our own move
                
                        // store the bestMove found uptill now
                        // NB this is not done in case of an AIStoppedException in alphaBeat()                
                    // print the results for debugging reasons                    
                    }            
        } catch (AIStoppedException ex) {  /* nothing to do */  }
        
        if (bestMove==null) {
            System.err.println("no valid move found!");
            return getRandomValidMove(s);
        } else {
            System.err.format(
                        "%s: depth= %2d, best move = %5s, value=%d\n", 
                        this.getClass().getSimpleName(),maxSearchDepth, bestMove, bestValue
                    );
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
        DraughtsState state = node.getState();
        
//        System.out.println("WhitePcs: " + whitePcs + " BlackPcs: " + blackPcs +" lastWhitePcs: " + lastwhitePcs + " lastBlackPcs: " + lastblackPcs);
//        System.out.println("Current Depth: " + curDepth + " Depth Limit: " + depth);
        if(curDepth >= depth && isQuite()){
            return evaluate(state);          
        }
        int bestVal = MAX_VALUE, i = 0;
        int bestMoveIndex = 0;
        List<Move> moves = state.getMoves();
//        boolean isLeaf = moves.isEmpty();
        if (moves.isEmpty()) {
            return evaluate(state);
        }
//        if (depth >= 10) {
//            moves = sortMoves(node, moves);            
//        }
        countPcs(state);
        lastwhiteCount = whiteCount;
        lastblackCount = blackCount;
        for (Move move : moves){           
            state.doMove(move);
            int value = alphaBetaMax(node, alpha, beta, depth, curDepth + 1);
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
        node.setBestMove(moves.get(bestMoveIndex));
        return bestVal;
     }
    
    int alphaBetaMax(DraughtsNode node, int alpha, int beta, int depth, int curDepth)
            throws AIStoppedException {
        if (stopped) { stopped = false; throw new AIStoppedException(); }
        DraughtsState state = node.getState();        
//        System.out.println("WhitePcs: " + whitePcs + " BlackPcs: " + blackPcs +" lastWhitePcs: " + lastwhitePcs + " lastBlackPcs: " + lastblackPcs);
//        System.out.println("Current Depth: " + curDepth + " Depth Limit: " + depth);
        if(curDepth >= depth && isQuite()){
            return evaluate(state);
        }
        int bestVal = MIN_VALUE, i = 0;
        int bestMoveIndex = 0;
        List<Move> moves = state.getMoves();
//        boolean isLeaf = moves.isEmpty();
        if (moves.isEmpty()) {
            return evaluate(state);
        }
//        if (depth >= 10){
//            moves = sortMoves(node, moves);
//            Collections.reverse(moves); //Compensate for the fact that Heapsort sorts from small to large
//        }
        countPcs(state);
        lastwhiteCount = whiteCount;
        lastblackCount = blackCount;
        for (Move move : moves){
            state.doMove(move);            
            int value = alphaBetaMin(node, alpha, beta, depth, curDepth + 1);
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
        node.setBestMove(moves.get(bestMoveIndex));
        return bestVal;
    }
    
    void countPcs (DraughtsState state){
        int[] pieces = state.getPieces();
        whiteCount = 0;
        blackCount = 0;
        whitePiece = 0;
        whiteKing = 0;
        blackPiece = 0;
        blackKing = 0;
        for (int i = 1; i < 51; i++){
            if (pieces[i] == DraughtsState.WHITEPIECE) {                
                whitePiece++;
            }
            else if (pieces[i] == DraughtsState.WHITEKING) {
                whiteKing++;
            }
            else if(pieces[i] == DraughtsState.BLACKPIECE) {
                blackPiece++;
            }
            else if(pieces[i] == DraughtsState.BLACKKING) {
                blackKing++;
            }
        }
        whiteCount = whitePiece + 3 * whiteKing;
        blackCount = blackPiece + 3 * blackKing;
    }
    
    boolean isQuite () {        
        if (whiteCount == lastwhiteCount || blackCount == lastblackCount){//Not the perfect condition for checking quiescence        
            return true;
        }
        else {
//            System.out.println("State is not quite");
            return false;
        }        
    }
    
//    List<Move> sortMoves(DraughtsNode node, List<Move> moves) {
//        DraughtsState state = node.getState();
//        int length = moves.size();
//        int[] evaluations = new int[length];
//        
////        List<Move> newMoves = new ArrayList<>(moves);
//        for (Move move : moves){
//            state.doMove(move);
//            countPcs(state);
//            DraughtsState new_state = node.getState();
//            evaluations[moves.indexOf(move)] = evaluate(new_state);
//            state.undoMove(move);            
//        }
//        //Build heap
//        for (int i = length / 2 - 1; i >= 0; i--) {
//            heapifyMoves(evaluations, moves, length, i);
//        }
//        //Extract one by one the largest element from the heap
//        for (int i = length - 1; i >= 0; i--){
//            //Exchange the root node and the end node
//            int temp = evaluations[0];
//            evaluations[0] = evaluations[i];
//            evaluations[i] = temp;
//            Move tmp_move = moves.get(0);
//            moves.set(0, moves.get(i));
//            moves.set(i, tmp_move);
//            //Maintain the heap properties for the remaining heap
//            heapifyMoves(evaluations, moves, i, 0);
//        }
//        
//        return moves;
//    }
//    
//    void heapifyMoves(int[] evals, List<Move> Moves, int n, int i) {
//        int largest = i;
//        int left = i * 2 + 1;
//        int right = i * 2 + 2;
//        
//        if (left < n && evals[left] > evals[largest]) {
//            largest = left;
//        }
//        if (right < n && evals[right] > evals[largest]) {
//            largest = right;
//        }
//        if(largest != i) {
//            int tmp = evals[i];
//            evals[i] = evals[largest];
//            evals[largest] = tmp;
//            Move tmp_move = Moves.get(i);
//            Moves.set(i, Moves.get(largest));
//            Moves.set(largest, tmp_move);
//            
//            heapifyMoves(evals, Moves, n, largest);
//        }
//    }
    
    /** A method that evaluates the given state. */
    int evaluate(DraughtsState state) { 
//        int[] pieces = state.getPieces();
        int material;
//        int tempi, whiteTempi = 0, blackTempi = 0;
//        int centring, whiteCentr = 0, blackCentr = 0;
//        int formation, whiteForm = 0, blackForm = 0;
//        int balance, whiteBalance = 0, blackBalance = 0;
        int matWeight = 6;//, tempiWeight = 2, centrWeight = 1, formWeight = 1, balanceWeight = 1;
//        for (int i = 1; i < 51; i++){
//            if (pieces[i] == DraughtsState.WHITEPIECE){
////                whiteCount++;
//                whiteTempi += -(Math.ceil(i / 5) - 10);
//                whiteCentr += -(Math.abs(i % 5 - 3) - 2);
////                whiteBalance += Integer.signum(Math.abs(i % 5 - 3));
//                whiteBalance += Integer.signum(i % 5 - 3);
//            } else if(pieces[i] == DraughtsState.WHITEKING){         
////                whiteCount += 3;
//                whiteTempi += 10;
//                whiteCentr += -(Math.abs(i % 5 - 3) - 2);
//            } else if(pieces[i] == DraughtsState.BLACKPIECE) {
////                blackCount++;
//                blackTempi += Math.ceil(i / 5);
//                blackCentr += -(Math.abs(i % 5 - 3) - 2);
////                blackBalance += Integer.signum(Math.abs(i % 5 - 3));
//                blackBalance += Integer.signum(i % 5 - 3);
//            } else if(pieces[i] == DraughtsState.BLACKKING) {      
////                blackCount += 3;
//                blackTempi += 10;
//                blackCentr += -(Math.abs(i % 5 - 3) - 2);
//            }
//            
//        }        
        material = whiteCount - blackCount;
//        tempi = whiteTempi - blackTempi;
//        centring = whiteCentr - blackCentr;
//        formation = whiteForm - blackForm;
//        balance = whiteBalance - blackBalance;
        
        int evaluation = matWeight * material;// + tempiWeight * tempi + centrWeight * centring + formWeight * formation + balanceWeight * balance;
        return evaluation; 
    }
}

