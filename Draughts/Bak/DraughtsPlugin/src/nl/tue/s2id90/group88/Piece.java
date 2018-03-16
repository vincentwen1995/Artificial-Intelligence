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
class Piece {


    boolean crowned;
    color c;
    int position;

    Piece(int index, color Color){
        position = index;
        crowned = false;
        c = Color;
    }
}

enum color{
    BLACK, WHITE;
};