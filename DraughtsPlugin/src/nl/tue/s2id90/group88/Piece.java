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
public class Piece {
    public enum colors{
        BLACK, WHITE;
    };

    boolean superState;
    public int x, y;
    public colors c;

    Piece(int inX, int inY, colors Color){
        x = inX;
        y = inY;
        superState = false;
        c = Color;
    }
}
