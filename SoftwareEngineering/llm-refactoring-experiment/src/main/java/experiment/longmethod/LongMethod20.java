package experiment.longmethod;

public class LongMethod20 {
    public double estimateEnergy(int hours, int devices, boolean peak, boolean batteryBackup) {
        double energy = hours * devices * 1.5;
        if (peak) {
            energy *= 1.25;
        } else {
            energy *= 0.9;
        }
        if (batteryBackup) {
            energy += 12;
        }
        if (devices > 10) {
            energy += devices / 2.0;
        }
        return energy;
    }
}
