package experiment.duplicatedcode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

public class DuplicateCode01Test {

    @Test
    void score_zero() {
        assertEquals(2 * (0 + 1), DuplicateCode01.score(0));
    }

    @Test
    void score_negative() {
        assertEquals(2 * (-1 + 1), DuplicateCode01.score(-1));
    }

    @Test
    void score_and_scoreAgain_match_for_values() {
        int[] values = {0, 1, 5, -2, 17};
        for (int v : values) {
            int expected = DuplicateCode01.score(v);
            assertEquals(expected, DuplicateCode01.scoreAgain(v));
        }
    }
}
