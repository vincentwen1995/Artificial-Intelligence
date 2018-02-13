/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.group88;
import java.util.*;
/**
 *
 * @author Michiel
 */
class GameState {
    public enum cellState{
        EMPTY, WHITE, BLACK
    };

    GameState(){
        List<Piece> blacks = new ArrayList<>(20);
        List<Piece> whites = new ArrayList<>(20);
        for(int i = 1; i < 21; i++){
            blacks.set(i-1, new Piece(i, color.BLACK));
        }
        for(int i = 26; i < 51; i++){
            whites.set(i-26, new Piece(i, color.WHITE));
        }
    }
    Move getMoves(){
        Move move = new Move();
        
        return move;
    }
    
    void doMove(Move move){
        move.piece.position = move.finalPosition;
        
    }
    
    void undoMove(Move move){
        move.piece.position = move.initialPosition;
    }
}
