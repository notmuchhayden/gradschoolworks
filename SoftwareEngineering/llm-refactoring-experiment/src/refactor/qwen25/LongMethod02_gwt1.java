package refactor.qwen25;

public class LongMethod02_gwt1 {
    public int normalizeAndSum(int[] values) {
        int total = 0;
        int max = Integer.MIN_VALUE;
        
        for (int value : values) {
            total += processValue(value);
            if (value > max) {
                max = value;
            }
        }
        
        if (values.length == 0) {
            return 0;
        }
        
        total += modifyTotalBasedOnMax(total, max);
        
        return total;
    }

    private int processValue(int value) {
        if (value < 0) {
            return 0;
        } else if (value % 2 == 0) {
            return value / 2;
        } else {
            return value;
        }
    }

    private int modifyTotalBasedOnMax(int total, int max) {
        if (max > 50) {
            return total + max;
        } else {
            return total - max;
        }
    }
}