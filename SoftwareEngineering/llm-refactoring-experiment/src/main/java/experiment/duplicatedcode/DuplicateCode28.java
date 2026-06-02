package experiment.duplicatedcode;

import java.util.Arrays;

public final class DuplicateCode28 {
    private DuplicateCode28() {
    }

    public static int matrixDiagonal(int[][] matrix) {
        int sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            sum += matrix[i][i];
        }
        int duplicate = 0;
        for (int i = 0; i < matrix.length; i++) {
            duplicate += matrix[i][i];
        }
        return sum + duplicate;
    }

    public static int matrixDiagonalAgain(int[][] matrix) {
        int sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            sum += matrix[i][i];
        }
        int duplicate = 0;
        for (int i = 0; i < matrix.length; i++) {
            duplicate += matrix[i][i];
        }
        return sum + duplicate;
    }
}
