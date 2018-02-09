/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.group88;

/**
 *
 * @author Michiel
 */
public class GameState {
    public enum cellState{
        EMPTY, WHITE, BLACK
    };
    static {
        cellState board[][] = new cellState[10][10]; //Lets waste memory space
        for(int x = 0; x < 10; x++){
            for(int y = 0; y < 4; y++){
                if(x%2 == 1 && y%2 == 1){
                    board[x][y] = cellState.BLACK;
                } else {
                    board[x][y] = cellState.EMPTY;
                }
            }
            for(int y = 6; y < 10; y++){
                if(x%2 == 1 && y%2 == 1){
                    board[x][y] = cellState.WHITE;
                } else {
                    board[x][y] = cellState.EMPTY;
                }
            }
            for(int y = 4; y < 6; y++){
                board[x][y] = cellState.EMPTY;
            }
        }
    }    
            
    public Move getMoves(){
        Move move = new Move();
        
        return move;
    }
}
