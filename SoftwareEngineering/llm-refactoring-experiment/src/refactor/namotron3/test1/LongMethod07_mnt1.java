package experiment.longmethod;

public class LongMethod07_mnt1 {

    public String classifyTemperature(double temp, double humidity, boolean cloudy) {
        String category = getTemperatureCategory(temp);
        category = applyHumidityModifier(category, humidity);
        category = applyWeatherSuffix(category, cloudy);
        return category;
    }

    private String getTemperatureCategory(double temp) {
        if (temp < 0) {
            return "freezing";
        } else if (temp < 10) {
            return "cold";
        } else if (temp < 25) {
            return "mild";
        } else {
            return "hot";
        }
    }

    private String applyHumidityModifier(String base, double humidity) {
        if (humidity > 80) {
            return base + "-humid";
        } else if (humidity < 30) {
            return base + "-dry";
        }
        return base;
    }

    private String applyWeatherSuffix(String base, boolean cloudy) {
        if (cloudy) {
            return base + "-cloudy";
        }
        return base;
    }
}