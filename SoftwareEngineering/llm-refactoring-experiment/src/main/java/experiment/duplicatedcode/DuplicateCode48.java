package experiment.duplicatedcode;

import java.util.ArrayList;
import java.util.List;

public final class DuplicateCode48 {
    private DuplicateCode48() {
    }

    record CellWindow(int row, int column, int value) {
        int score() {
            return row + column + value;
        }
    }

    public static List<Integer> summarize(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < matrix[row].length; column++) {
                result.add(new CellWindow(row, column, matrix[row][column]).score());
            }
        }
        List<Integer> duplicate = new ArrayList<>();
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < matrix[row].length; column++) {
                duplicate.add(new CellWindow(row, column, matrix[row][column]).score());
            }
        }
        result.addAll(duplicate);
        return result;
    }

    public static List<Integer> summarizeAgain(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < matrix[row].length; column++) {
                result.add(new CellWindow(row, column, matrix[row][column]).score());
            }
        }
        List<Integer> duplicate = new ArrayList<>();
        for (int row = 0; row < matrix.length; row++) {
            for (int column = 0; column < matrix[row].length; column++) {
                duplicate.add(new CellWindow(row, column, matrix[row][column]).score());
            }
        }
        result.addAll(duplicate);
        return result;
    }
}
