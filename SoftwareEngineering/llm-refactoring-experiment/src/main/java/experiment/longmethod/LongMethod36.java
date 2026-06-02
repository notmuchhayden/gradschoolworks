package experiment.longmethod;

public class LongMethod36 {
    public double computeVolume(double length, double width, double height, boolean roundUp) {
        double volume = length * width * height;
        if (roundUp) {
            volume = Math.ceil(volume);
        } else {
            volume = Math.floor(volume);
        }
        if (volume > 1000) {
            volume += 10;
        }
        return volume;
    }
}
